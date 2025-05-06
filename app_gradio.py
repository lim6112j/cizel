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
    # With type="messages", history is List[Dict[str, str | List[Tuple[str|None, str|None]] | None]]
    # Each dict is like: {"role": "user", "content": "Hi"} or
    # {"role": "assistant", "content": "Hello"} or
    # {"role": "assistant", "content": [("path/to/image.png", "alt_text_or_caption")]}
    langgraph_messages_history: list[BaseMessage] = []
    for message_dict in history:
        role = message_dict.get("role")
        content = message_dict.get("content")

        if role == "user" and isinstance(content, str):
            langgraph_messages_history.append(HumanMessage(content=content))
        elif role == "assistant":
            text_for_ai_message = None
            if isinstance(content, str): # Assistant's response was pure text
                text_for_ai_message = content
            elif isinstance(content, list) and content: # Assistant's response included an image
                # content is like [(filepath, alt_text_or_None)]
                # The alt_text is the textual part of the bot's response for that turn.
                first_element = content[0] # Assuming one primary (image, caption) tuple per turn
                if isinstance(first_element, tuple) and len(first_element) > 1 and first_element[1]:
                    text_for_ai_message = str(first_element[1])
                # If no alt_text (first_element[1] is None), text_for_ai_message remains None.
                # This should be handled by ensuring our function always returns text with an image.
            
            if text_for_ai_message: # Only add AIMessage if we have text content
                langgraph_messages_history.append(AIMessage(content=text_for_ai_message))

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

    # 5. Return the formatted response for Gradio ChatInterface (type="messages" mode)
    # The function should return the *content* for the assistant's message.
    # For text, it's a string. For files, let's try List[Tuple[str, str | None]].
    if image_path_for_turn and text_response_for_turn:
        # Image and text: Return list with tuple (filepath, caption)
        return [(image_path_for_turn, text_response_for_turn)]
    elif image_path_for_turn: # Only image was produced, text_response_for_turn is empty
        # Extract the tool message content as the caption
        tool_message_text = "Image generated." # Default caption
        for msg in new_messages_from_graph:
            if isinstance(msg, ToolMessage) and msg.name == "midjourney_image_generator":
                tool_message_text = str(msg.content) # Use tool success message as caption
                break
        # Image only: Return list with tuple (filepath, caption)
        return [(image_path_for_turn, tool_message_text)]
    elif text_response_for_turn:
        # Text only: Return the string content directly
        return text_response_for_turn
    else:
        # Fallback if no discernible output, though agent should always respond.
        # This might happen if the agent ends without a final AIMessage to the user.
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
        render=False, # Render manually for layout options if needed, but fine for ChatInterface
        type="messages" # Use OpenAI-style message format
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
