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

    # 5. Return the formatted response for Gradio
    # For Gradio 3.x compatibility, we need to return either a string or a tuple (image_path, caption)
    if image_path_for_turn:
        caption_text = text_response_for_turn or "Generated image"
        # Return a tuple for image responses
        return (image_path_for_turn, caption_text)
    else:
        # Return a string for text-only responses
        return text_response_for_turn or "Agent processed the request."


# Create a custom interface instead of using ChatInterface
with gr.Blocks() as iface:
    gr.Markdown("# LangGraph Agent Chat")
    gr.Markdown("Chat with the AI agent. It can generate images (e.g., 'draw a picture of a happy dog') or answer questions.")
    
    chatbot = gr.Chatbot(
        height=600,
        label="Conversation",
        show_label=True,
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here...",
            scale=7,
            show_label=False,
        )
        submit = gr.Button("Send", variant="primary")
    
    # Add example buttons
    with gr.Accordion("Examples", open=False):
        examples = gr.Examples(
            examples=[
                "Generate an image of a futuristic city at sunset",
                "What is LangGraph?",
                "Make a picture of a cat programming on a laptop"
            ],
            inputs=msg
        )
    
    # Define the chat function
    def respond(message, chat_history):
        if not message:
            return "", chat_history
        
        # Add user message to history
        chat_history.append((message, None))
        
        # Get response from agent
        response = chat_with_agent(message, chat_history[:-1])
        
        # Update the last message with the response
        chat_history[-1] = (message, response)
        
        return "", chat_history
    
    # Set up event handlers
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    # Check for necessary API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not found. Please set it in your .env file or environment.")
    # STABILITY_API_KEY is checked by the tool itself, but good to remind here.
    if not os.getenv("STABILITY_API_KEY"):
        print("⚠️ STABILITY_API_KEY not found. Image generation will fail. Please set it in your .env file or environment.")
    
    print("🚀 Launching Gradio interface...")
    iface.launch()
