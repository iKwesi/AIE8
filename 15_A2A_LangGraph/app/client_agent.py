"""Simple LangGraph client agent that uses the A2A server agent as a tool."""
from __future__ import annotations

import os
from typing import Annotated, TypedDict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from app.a2a_tool import call_a2a_agent


class ClientAgentState(TypedDict):
    """State schema for the client agent."""
    messages: Annotated[List[BaseMessage], add_messages]


def build_client_agent():
    """Build a simple LangGraph client agent that uses the A2A server.
    
    This agent has a single tool: call_a2a_agent, which delegates queries
    to the A2A server agent via the A2A protocol.
    
    Returns:
        Compiled LangGraph agent
    """
    # Initialize the LLM
    model = ChatOpenAI(
        model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0,
    )
    
    # Bind the A2A tool to the model
    model_with_tools = model.bind_tools([call_a2a_agent])
    
    # Define the agent node
    def call_model(state: ClientAgentState):
        """Invoke the model with the current messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Define routing logic
    def should_continue(state: ClientAgentState):
        """Determine if we should call tools or end."""
        last_message = state["messages"][-1]
        # If there are tool calls, continue to the action node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "action"
        # Otherwise, end
        return "end"
    
    # Build the graph
    graph = StateGraph(ClientAgentState)
    
    # Add nodes
    graph.add_node("agent", call_model)
    graph.add_node("action", ToolNode([call_a2a_agent]))
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",
            "end": END
        }
    )
    graph.add_edge("action", "agent")
    
    # Compile with memory for multi-turn conversations
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_client_agent(query: str, thread_id: str = "default"):
    """Run the client agent with a query.
    
    Args:
        query: The user's question or request
        thread_id: Thread ID for conversation context (default: "default")
        
    Returns:
        The agent's final response
    """
    agent = build_client_agent()
    
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [("user", query)]}
    
    # Stream the agent's execution
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    for event in agent.stream(inputs, config, stream_mode="values"):
        # Get the last message
        last_message = event["messages"][-1]
        
        # Print tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("🔧 Calling A2A server agent...")
            
        # Print tool responses
        if hasattr(last_message, "content") and hasattr(last_message, "name"):
            if last_message.name == "call_a2a_agent":
                print(f"\n📥 Response from A2A server:\n")
                print(last_message.content)
                print()
    
    # Get final state
    final_state = agent.get_state(config)
    final_message = final_state.values["messages"][-1]
    
    return final_message.content


async def run_client_agent_async(query: str, thread_id: str = "default"):
    """Async version of run_client_agent.
    
    Args:
        query: The user's question or request
        thread_id: Thread ID for conversation context (default: "default")
        
    Returns:
        The agent's final response
    """
    agent = build_client_agent()
    
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [("user", query)]}
    
    # Stream the agent's execution
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    async for event in agent.astream(inputs, config, stream_mode="values"):
        # Get the last message
        last_message = event["messages"][-1]
        
        # Print tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("🔧 Calling A2A server agent...")
            
        # Print tool responses
        if hasattr(last_message, "content") and hasattr(last_message, "name"):
            if last_message.name == "call_a2a_agent":
                print(f"\n📥 Response from A2A server:\n")
                print(last_message.content)
                print()
    
    # Get final state
    final_state = await agent.aget_state(config)
    final_message = final_state.values["messages"][-1]
    
    return final_message.content
