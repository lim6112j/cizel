from langgraph_project import app
from PIL import Image
import io
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# It's good practice to load .env file at the start of your application
load_dotenv()


# Import the compiled graph app from our project


def generate_graph_diagram(output_path="workflow_diagram.png"):
    """Generates a PNG diagram of the LangGraph workflow."""
    try:
        print(f"\n📊 Generating workflow diagram at '{output_path}'...")
        # Get the graph object
        graph = app.get_graph()
        # Draw the graph to a PNG byte stream
        png_bytes = graph.draw_mermaid_png()
        # Save the bytes to a file
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"✅ Diagram saved successfully.")
    except ImportError as e:
        print(f"⚠️ Error generating diagram: {e}")
        print("   Please ensure 'pygraphviz' and 'matplotlib' are installed.")
        print("   On macOS, you might need: brew install graphviz")
        print("   Then run: uv pip install pygraphviz matplotlib pillow")
    except Exception as e:
        print(
            f"⚠️ An unexpected error occurred during diagram generation: {e}")


def run_agent(query: str):
    """Runs the LangGraph agent with the given query."""
    print(f"\n🚀 Starting agent with query: \"{query}\"")

    inputs = {"messages": [HumanMessage(content=query)]}

    # You can stream events for more detailed logging
    # for event in app.stream(inputs, {"recursion_limit": 10}):
    #     for key, value in event.items():
    #         print(f"Node: {key}")
    #         print(f"Value: {value}")
    #         print("---")
    # print("--- End of Stream ---")

    # Or invoke to get the final state
    final_state = app.invoke(inputs, {"recursion_limit": 10})

    print("\n✅ Agent finished. Final conversation state:")
    for message in final_state['messages']:
        message_type = message.type.upper()
        content = message.content

        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls_str = ", ".join(
                [f"{tc['name']}({tc['args']})" for tc in message.tool_calls])
            print(f"[{message_type}] {content} (Tool Calls: {tool_calls_str})")
        else:
            print(f"[{message_type}] {content}")


if __name__ == "__main__":
    # Check for essential environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with OPENAI_API_KEY='your_key_here'")
        exit(1)

    # Example 1: Using the Midjourney tool
    image_prompt = "Generate an image of a futuristic city at sunset using Midjourney. with text as 'kim ja tae' in the middle of image"
    run_agent(image_prompt)

    # Example 2: A general query not requiring tools
    general_query = "What is LangGraph?"
    run_agent(general_query)

    # Generate the workflow diagram
    # generate_graph_diagram()
