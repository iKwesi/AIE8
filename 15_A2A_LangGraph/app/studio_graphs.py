"""Wrapper functions for LangGraph Studio compatibility.

These wrappers adapt the existing graph builders to match LangGraph Studio's
expected signature (single RunnableConfig parameter).
"""
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
import os

from app.agent_graph_with_helpfulness import build_agent_graph_with_helpfulness
from app.client_agent import build_client_agent as _build_client_agent


def build_server_agent_for_studio(config: RunnableConfig):
    """Wrapper for server agent graph compatible with LangGraph Studio.
    
    This function adapts build_agent_graph_with_helpfulness to match
    the signature expected by LangGraph Studio.
    
    Args:
        config: RunnableConfig (required by LangGraph Studio, not used here)
        
    Returns:
        Compiled LangGraph agent with helpfulness evaluation
    """
    # Initialize model
    model = ChatOpenAI(
        model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0,
    )
    
    # System and format instructions
    system_instruction = (
        'You are a helpful AI assistant with access to various tools including web search, '
        'academic paper search, and document retrieval. '
        'Use the appropriate tools to answer user questions accurately and thoroughly. '
        'If you cannot find relevant information using the available tools, '
        'clearly state that you were unable to find the requested information.'
    )
    
    format_instruction = (
        'Set response status to input_required if the user needs to provide more information to complete the request. '
        'Set response status to error if there is an error while processing the request. '
        'Set response status to completed if the request is complete.'
    )
    
    # Build and return the graph
    return build_agent_graph_with_helpfulness(
        model=model,
        system_instruction=system_instruction,
        format_instruction=format_instruction,
        checkpointer=None  # Studio manages checkpointing
    )


def build_client_agent_for_studio(config: RunnableConfig):
    """Wrapper for client agent graph compatible with LangGraph Studio.
    
    This function adapts build_client_agent to match the signature
    expected by LangGraph Studio.
    
    Args:
        config: RunnableConfig (required by LangGraph Studio, not used here)
        
    Returns:
        Compiled LangGraph client agent
    """
    return _build_client_agent()
