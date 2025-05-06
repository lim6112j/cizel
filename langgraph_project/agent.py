import os
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from dotenv import load_dotenv

from .tools import midjourney_image_generator

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# 1. Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 2. Initialize Tools
# Add any other tools you want your agent to use here
tools = [midjourney_image_generator]
tool_executor = ToolExecutor(tools)

# 3. Initialize Model
# Ensure your OPENAI_API_KEY is set in your .env file or environment
# You can replace ChatOpenAI with any other LangChain compatible model
model = ChatOpenAI(temperature=0, streaming=True)
# Bind tools to the model for it to learn how to call them
model = model.bind_tools(tools)

# 4. Define Agent Logic Nodes

def should_continue(state: AgentState) -> str:
    """Determines the next step: call tools or end."""
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue_to_tools"
    return "end_conversation"

def call_model_node(state: AgentState) -> dict:
    """Invokes the LLM to get a response or tool calls."""
    print("🧠 Calling model...")
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

def call_tools_node(state: AgentState) -> dict:
    """Executes the tools called by the LLM."""
    print("🛠️ Calling tools...")
    last_message = state['messages'][-1]  # This will be an AIMessage with tool_calls
    
    actions = []
    # Ensure tool_calls is not None and is iterable
    original_tool_calls = last_message.tool_calls if last_message.tool_calls else []

    for tool_call in original_tool_calls:
        actions.append(ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        ))
    
    responses = tool_executor.batch(actions)
    
    tool_messages = []
    for i, response_content in enumerate(responses):
        tool_messages.append(ToolMessage(
            content=str(response_content),
            name=original_tool_calls[i]["name"],
            tool_call_id=original_tool_calls[i]["id"]
        ))
        
    return {"messages": tool_messages}

# 5. Define the Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model_node)
workflow.add_node("tools_action", call_tools_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue_to_tools": "tools_action",
        "end_conversation": END,
    },
)

workflow.add_edge("tools_action", "agent")

# Compile the graph into a runnable
app = workflow.compile()
