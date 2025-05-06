# LangGraph Midjourney Project

This project demonstrates a simple LangGraph agent capable of using a (mock) Midjourney tool to generate images, alongside general conversational abilities powered by an LLM.

## Setup

1.  **Create Project Files:**
    Ensure all project files (`pyproject.toml`, `run.py`, and the `langgraph_project` directory with its contents) are created as specified.

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    This project uses Poetry for dependency management.
    ```bash
    pip install poetry
    poetry install
    ```
    This will install all dependencies listed in `pyproject.toml`.

4.  **Set Up Environment Variables:**
    Create a `.env` file in the root directory of the project:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    # If your actual Midjourney tool requires an API key:
    # MIDJOURNEY_API_KEY="your_midjourney_api_key_here"
    ```
    Replace `"your_openai_api_key_here"` with your actual OpenAI API key.

## Running the Agent

Once the setup is complete, you can run the agent using the `run.py` script:

```bash
python run.py
```

The script will execute two predefined queries:
1.  A request to generate an image, which should trigger the Midjourney tool.
2.  A general question, which the LLM should answer directly.

## Project Structure

-   `pyproject.toml`: Defines project metadata and dependencies for Poetry.
-   `run.py`: The main script to execute the LangGraph agent.
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
