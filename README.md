# LangGraph Midjourney Project

This project demonstrates a simple LangGraph agent capable of using a (mock) Midjourney tool to generate images, alongside general conversational abilities powered by an LLM.

## Setup

1.  **Create Project Files:**
    Ensure all project files (`pyproject.toml`, `run.py`, and the `langgraph_project` directory with its contents) are created as specified.

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    This project uses `uv` for fast dependency management. If you don't have `uv`, install it first (see https://github.com/astral-sh/uv).
    ```bash
    # Create and activate a virtual environment (recommended)
    python3 -m venv venv  # Or use: uv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install dependencies using uv
    # It's recommended to compile a lock file first
    uv pip compile pyproject.toml -o requirements.lock
    uv pip sync requirements.lock
    # Or install directly from pyproject.toml (might be slower)
    # uv pip install .

    # Note: pygraphviz requires the graphviz library.
    # On macOS: brew install graphviz
    # On Debian/Ubuntu: sudo apt-get install graphviz graphviz-dev
    # On Fedora: sudo dnf install graphviz graphviz-devel
    ```
    This will install all dependencies listed in `pyproject.toml`, including those needed for diagram generation (`pygraphviz`, `matplotlib`, `pillow`), the Stability AI tool (`requests`), and the web interface (`gradio`).

4.  **Set Up Environment Variables:**
    Create a `.env` file in the root directory of the project:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    # The image generation tool now uses Stability AI:
    STABILITY_API_KEY="your_stability_ai_api_key_here"
    ```
    Replace the placeholder keys with your actual API keys. You can get a Stability AI key from their website.

## Running the Agent

This project offers two ways to interact with the agent:

### 1. Web Chat Interface (Recommended)

Once the setup is complete, you can start the Gradio web interface:

```bash
python app_gradio.py
```

This will launch a local web server (usually at `http://127.0.0.1:7860` or similar), which you can open in your browser to chat with the agent. The interface supports text conversation and image display if the agent generates an image.

### 2. Command-Line Interface

You can also run the agent using the `run.py` script for basic command-line interaction:

```bash
python run.py
```

The `run.py` script will execute two predefined queries:
1.  A request to generate an image.
2.  A general question.

The `run.py` script can also generate a `workflow_diagram.png` file (if uncommented) visualizing the LangGraph agent's structure.

## Project Structure

-   `pyproject.toml`: Defines project metadata and dependencies (PEP 621 format).
-   `app_gradio.py`: Script to launch the Gradio web chat interface.
-   `run.py`: Script for command-line execution of the LangGraph agent.
-   `langgraph_project/`: The core Python package for the LangGraph application.
    -   `__init__.py`: Initializes the `langgraph_project` package and exports the `app`.
    -   `agent.py`: Contains the definition of the agent's state, tools, model, nodes, and the graph itself.
    -   `tools/`: A sub-package for custom tools.
        -   `__init__.py`: Initializes the `tools` sub-package.
        -   `midjourney.py`: Implements the mock Midjourney tool.
-   `.env` (to be created by you): Stores sensitive information like API keys.
-   `README.md`: This file, providing an overview and instructions.

## Customization

-   **Tools**: Add more tools to the `langgraph_project/tools/` directory, import them in `agent.py`, and add them to the `tools` list.
-   **Model**: Change the LLM by modifying the `model` initialization in `agent.py` (e.g., use a different provider like Anthropic or a local model).
-   **Agent Logic**: Modify the nodes and edges in `agent.py` to change the agent's behavior and control flow.
# cizel
