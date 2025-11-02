"""CrewAI-compatible wrapper for the A2A tool.

This module provides a CrewAI-compatible version of the A2A tool
so that CrewAI agents can use it. Uses ThreadPoolExecutor to run async
A2A SDK code in a separate thread to avoid event loop conflicts.
"""
import asyncio
from typing import Type
from concurrent.futures import ThreadPoolExecutor
import httpx
from uuid import uuid4
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest


class A2AToolInput(BaseModel):
    """Input schema for A2A tool."""
    query: str = Field(..., description="The question or request to send to the A2A agent")


class A2ACrewTool(BaseTool):
    """CrewAI tool for calling the A2A server agent.
    
    This tool connects to an A2A-compliant agent server and delegates queries to it.
    The server has access to:
    - Web search (Tavily) for current information
    - Academic paper search (ArXiv) for research papers
    - Document retrieval (RAG) for internal documents
    
    Uses ThreadPoolExecutor to run async A2A SDK code in a separate thread,
    avoiding event loop conflicts with CrewAI's async execution.
    """
    
    name: str = "call_a2a_agent"
    description: str = (
        "Call the A2A server agent to answer questions using web search, "
        "academic papers, or document retrieval. Use this tool to delegate "
        "complex queries to a specialized agent with multiple capabilities."
    )
    args_schema: Type[BaseModel] = A2AToolInput
    
    def _run(self, query: str) -> str:
        """Execute the tool to call the A2A agent.
        
        This method is synchronous (required by CrewAI) but calls async A2A SDK
        code in a separate thread with its own event loop.
        
        Args:
            query: The question or request to send to the A2A agent
            
        Returns:
            The agent's response as a string
        """
        
        async def async_a2a_call():
            """Async function that uses the A2A SDK properly."""
            base_url = 'http://localhost:10000'
            
            try:
                # Use async httpx client
                async with httpx.AsyncClient(timeout=60.0) as httpx_client:
                    # Fetch agent card asynchronously
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
                    
                    # Send message and get response asynchronously
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
                return (
                    f"Could not connect to A2A server at {base_url}. "
                    "Make sure the server is running with: make server"
                )
            except httpx.TimeoutException:
                return "Request to A2A server timed out. The agent may be processing a complex query."
            except Exception as e:
                return f"Error calling A2A agent: {str(e)}"
        
        # Run the async function in a separate thread with its own event loop
        with ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, async_a2a_call())
            return future.result()


# Create instance to use in CrewAI agents
crewai_call_a2a_agent = A2ACrewTool()
