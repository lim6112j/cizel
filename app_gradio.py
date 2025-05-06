import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph_project import app as langgraph_app # Your compiled LangGraph app
import os
import re

# Ensure API keys are loaded (if not already handled by langgraph_project)
from dotenv import load_dotenv
load_dotenv()

def extract_image_path_from_tool_message(message_content: str) -> str | None:
    """
    Extracts a file path from the image generator tool's success message.
    Example message: "Successfully generated image for 'prompt'. Saved as: ./generated_image_prompt.webp"
    """
    match = re.search(r"Saved as: (\S+\.(?:webp|png|jpg|jpeg))", message_content)
    if match:
        path = match.group(1)
        # Check if the file exists before returning the path
        if os.path.exists(path):
            return path
        else:
            print(f"⚠️ Image path found in message, but file does not exist: {path}")
    return None

def chat_with_agent(user_input: str, history: list[list[str | tuple | None]]):
    """
    Handles a single turn of conversation with the LangGraph agent via Gradio.
    """
    # 1. Convert Gradio history and current user input to LangGraph message format
    # Assuming Gradio 3.x style history: List[Tuple[user_msg_str | None, bot_msg_representation | None]]
    langgraph_messages_history: list[BaseMessage] = []
    for user_msg_str, bot_msg_representation in history:
        if user_msg_str: # User message
            langgraph_messages_history.append(HumanMessage(content=user_msg_str))
        
        # Process bot message for history
        text_content_for_history = None
        if isinstance(bot_msg_representation, str):
            # Bot response was plain text
            text_content_for_history = bot_msg_representation
        elif isinstance(bot_msg_representation, tuple) and len(bot_msg_representation) == 2:
            # Bot response was likely an image tuple (filepath, alt_text)
            # Use the alt_text as the content for AIMessage history
            text_content_for_history = str(bot_msg_representation[1]) if bot_msg_representation[1] else "Image"
        
        if text_content_for_history:
            langgraph_messages_history.append(AIMessage(content=text_content_for_history))
        else:
            # If assistant message had no text (e.g., only image with no caption extracted)
            # We might still want to represent it, or decide the LLM doesn't need it.
            # For now, we only add AIMessages with text content.
            # The 'content' variable might not be defined here if bot_msg_representation was None or not a tuple/str
            # Let's print bot_msg_representation to see what it was.
            print(f"⚠️ Assistant message in history had no extractable text content. Bot representation: {bot_msg_representation}")

    current_turn_human_message = HumanMessage(content=user_input)
    complete_input_messages = langgraph_messages_history + [current_turn_human_message]
    
    num_messages_before_invoke = len(complete_input_messages)

    # 2. Invoke the LangGraph app
    # The agent.py expects the full message history for its state.
    final_state = langgraph_app.invoke({"messages": complete_input_messages}, {"recursion_limit": 15})
    
    # 3. Extract new messages produced by the agent in this turn
    all_graph_messages = final_state.get('messages', [])
    # New messages are those appended after our input sequence
    new_messages_from_graph = all_graph_messages[num_messages_before_invoke:]

    # 4. Format the response for Gradio
    # We want to construct a single bot response for this turn, which might include an image and text.
    image_path_for_turn = None
    text_response_for_turn = ""

    for msg in new_messages_from_graph:
        if isinstance(msg, AIMessage):
            # Concatenate all AIMessage content as the textual response
            # Usually, the final AIMessage contains the consolidated response.
            text_response_for_turn = msg.content # Overwrite with later AIMessages if they exist
        elif isinstance(msg, ToolMessage) and msg.name == "midjourney_image_generator":
            # Check for image path from tool message content
            path = extract_image_path_from_tool_message(str(msg.content))
            if path:
                image_path_for_turn = path
    
    text_response_for_turn = text_response_for_turn.strip()

    # 5. Return the formatted response for Gradio (assuming 3.x style ChatInterface)
    #    - String for text-only response.
    #    - Tuple (filepath, alt_text) for image response.

    if image_path_for_turn: # A valid image path was extracted
        # Determine the caption for the image.
        # Prefer LLM's direct response if available, otherwise use tool's message.
        caption_text = text_response_for_turn
        if not caption_text: # LLM provided no text, use tool's output message
            for msg_item in new_messages_from_graph:
                if isinstance(msg_item, ToolMessage) and msg_item.name == "midjourney_image_generator":
                    caption_text = str(msg_item.content) # This is "Successfully generated... Saved as: <path>"
                    break
        if not caption_text: # Fallback caption
             caption_text = "Generated image"
        
        return (image_path_for_turn, caption_text) # Return as (filepath, alt_text) tuple
    
    elif text_response_for_turn: # No image, but there is text from the LLM
        return text_response_for_turn
    
    else: # No image and no specific text from LLM (e.g., tool failed and LLM was silent)
        # Check if the last message from the graph was a ToolMessage (likely an error message from the tool)
        if new_messages_from_graph and isinstance(new_messages_from_graph[-1], ToolMessage):
            return str(new_messages_from_graph[-1].content) # Return tool's error message
        
        return "Agent processed the request. (No specific text or image output for this turn)"


# Create the Gradio interface using gr.ChatInterface
iface = gr.ChatInterface(
    fn=chat_with_agent,
    title="LangGraph Agent Chat",
    description="Chat with the AI agent. It can generate images (e.g., 'draw a picture of a happy dog') or answer questions.",
    examples=[
        ["Generate an image of a futuristic city at sunset"],
        ["What is LangGraph?"],
        ["Make a picture of a cat programming on a laptop"]
    ],
    chatbot=gr.Chatbot(
        height=600,
        label="Conversation",
        show_label=True,
        render=False # Render manually for layout options if needed, but fine for ChatInterface
        # Removed type="messages" for Gradio 3.x compatibility; defaults to "tuples" or similar
    ),
    textbox=gr.Textbox(
        placeholder="Type your message here...",
        container=False, # Part of ChatInterface's internal layout
        scale=7 # Relative width
    ),
    # retry_btn, undo_btn, clear_btn are typically default or managed by Chatbot in newer Gradio
    # theme="gradio/soft" # Optional: explore themes
)

if __name__ == "__main__":
    # Check for necessary API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not found. Please set it in your .env file or environment.")
    # STABILITY_API_KEY is checked by the tool itself, but good to remind here.
    if not os.getenv("STABILITY_API_KEY"):
        print("⚠️ STABILITY_API_KEY not found. Image generation will fail. Please set it in your .env file or environment.")
    
    print("🚀 Launching Gradio interface...")
    iface.launch()
