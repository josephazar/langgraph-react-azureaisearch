"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast, Any

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model, extract_sources_from_results

# Define the function that calls the model


async def call_model(state: State) -> Dict[str, Any]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.
    """
    configuration = Configuration.from_context()

    # Initialize the model with tool binding
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }
    
    # Check if the response should include sources
    if not response.tool_calls and state.search_results:
        # Check if the response is from general knowledge (not from knowledge base)
        is_general_knowledge = (
            "not from the knowledge base" in response.content.lower() or
            "from my general knowledge" in response.content.lower() or
            "NOTE:" in response.content
        )
        
        # Only add sources if:
        # 1. We have search results
        # 2. The response is NOT from general knowledge
        # 3. Sources aren't already in the response
        if not is_general_knowledge and "**Sources:**" not in response.content:
            sources = extract_sources_from_results(state.search_results)
            if sources:
                sources_text = "\n\n**Sources:**\n" + "\n".join(f"- {source}" for source in sources)
                response.content += sources_text
            
        # Clear search results after using them
        return {
            "messages": [response],
            "search_results": [],
            "sources_used": []
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Custom tool node that updates search results in state
async def tool_node(state: State) -> Dict[str, Any]:
    """Execute tools and update state with search results."""
    tool_node_instance = ToolNode(TOOLS)
    result = await tool_node_instance.ainvoke(state)
    
    # Check if the last message is a tool message with search results
    if result["messages"]:
        last_msg = result["messages"][-1]
        if hasattr(last_msg, 'content') and isinstance(last_msg.content, str):
            try:
                # Try to parse the tool response
                import json
                content = json.loads(last_msg.content) if last_msg.content.startswith('{') else {"content": last_msg.content}
                
                # If this was a search tool call, update search results
                if "results" in content:
                    return {
                        "messages": result["messages"],
                        "search_results": content["results"],
                        "sources_used": extract_sources_from_results(content["results"])
                    }
            except:
                pass
    
    return result

# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", tool_node)

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
