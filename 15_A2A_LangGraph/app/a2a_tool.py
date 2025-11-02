"""A2A API Tool - Enables a LangGraph agent to call another agent via A2A protocol."""
from __future__ import annotations

import httpx
from typing import Annotated
from langchain_core.tools import tool
from uuid import uuid4

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest


class A2AToolError(Exception):
    """Exception raised when A2A tool encounters an error."""
    pass


@tool
async def call_a2a_agent(
    query: Annotated[str, "The question or request to send to the A2A agent"]
) -> str:
    """Call the A2A server agent to answer questions using web search, academic papers, or document retrieval.
    
    This tool connects to an A2A-compliant agent server and delegates the query to it.
    The server has access to:
    - Web search (Tavily) for current information
    - Academic paper search (ArXiv) for research papers
    - Document retrieval (RAG) for internal documents
    
    Args:
        query: The user's question or request
        
    Returns:
        The agent's response as a string
        
    Raises:
        A2AToolError: If the server is unavailable or returns an error
    """
    base_url = 'http://localhost:10000'
    
    try:
        # Use longer timeout for LLM responses
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
            # Fetch agent card
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=base_url,
            )
            
            agent_card = await resolver.get_agent_card()
            
            # Initialize A2A client
            client = A2AClient(
                httpx_client=httpx_client,
                agent_card=agent_card
            )
            
            # Prepare message
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': query}
                    ],
                    'message_id': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload)
            )
            
            # Send message and get response
            response = await client.send_message(request)
            
            # Extract the response content
            if response.root and response.root.result:
                result = response.root.result
                
                # Get the last message from artifacts or messages
                if hasattr(result, 'artifacts') and result.artifacts:
                    # Extract text from artifacts
                    for artifact in result.artifacts:
                        if hasattr(artifact, 'parts') and artifact.parts:
                            for part in artifact.parts:
                                if hasattr(part.root, 'text'):
                                    return part.root.text
                
                # Fallback to messages if no artifacts
                if hasattr(result, 'messages') and result.messages:
                    last_message = result.messages[-1]
                    if hasattr(last_message, 'parts') and last_message.parts:
                        for part in last_message.parts:
                            if hasattr(part.root, 'text'):
                                return part.root.text
                
                return "The agent processed the request but returned no text content."
            
            return "No response received from the agent."
            
    except httpx.ConnectError:
        raise A2AToolError(
            f"Could not connect to A2A server at {base_url}. "
            "Make sure the server is running with: uv run python -m app"
        )
    except httpx.TimeoutException:
        raise A2AToolError(
            "Request to A2A server timed out. The agent may be processing a complex query."
        )
    except Exception as e:
        raise A2AToolError(f"Error calling A2A agent: {str(e)}")
