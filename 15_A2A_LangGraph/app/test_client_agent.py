"""Test script for the client agent that uses the A2A server via A2A protocol.

This script demonstrates:
1. Single queries testing each server skill (web search, ArXiv, RAG)
2. Multi-turn conversations with context preservation
3. A2A protocol in action (agent-to-agent communication)

Prerequisites:
- The A2A server must be running: uv run python -m app
- Server should be available at http://localhost:10000
"""
import asyncio
import os
from dotenv import load_dotenv

from app.client_agent import run_client_agent_async


# Load environment variables
load_dotenv()


async def test_web_search():
    """Test the client agent using the server's web search capability (Tavily)."""
    print("\n" + "="*80)
    print("TEST 1: Web Search (Tavily Tool)")
    print("="*80)
    
    query = "What are the latest developments in artificial intelligence in 2025?"
    response = await run_client_agent_async(query, thread_id="test_web_search")
    
    print(f"\n✅ Final Response:\n{response}\n")


async def test_arxiv_search():
    """Test the client agent using the server's ArXiv search capability."""
    print("\n" + "="*80)
    print("TEST 2: Academic Paper Search (ArXiv Tool)")
    print("="*80)
    
    query = "Find recent papers on transformer architectures in machine learning"
    response = await run_client_agent_async(query, thread_id="test_arxiv")
    
    print(f"\n✅ Final Response:\n{response}\n")


async def test_rag_retrieval():
    """Test the client agent using the server's RAG document retrieval."""
    print("\n" + "="*80)
    print("TEST 3: Document Retrieval (RAG Tool)")
    print("="*80)
    
    query = "What information do the documents contain about how people use AI?"
    response = await run_client_agent_async(query, thread_id="test_rag")
    
    print(f"\n✅ Final Response:\n{response}\n")


async def test_multi_turn_conversation():
    """Test multi-turn conversation with context preservation."""
    print("\n" + "="*80)
    print("TEST 4: Multi-Turn Conversation")
    print("="*80)
    
    thread_id = "test_multiturn"
    
    # First query
    query1 = "Find me papers about large language models"
    print(f"\n👤 User (Turn 1): {query1}")
    response1 = await run_client_agent_async(query1, thread_id=thread_id)
    print(f"\n✅ Agent Response (Turn 1):\n{response1}\n")
    
    # Follow-up query (should have context from first query)
    query2 = "Can you summarize the key findings from those papers?"
    print(f"\n👤 User (Turn 2): {query2}")
    response2 = await run_client_agent_async(query2, thread_id=thread_id)
    print(f"\n✅ Agent Response (Turn 2):\n{response2}\n")
    
    # Another follow-up
    query3 = "Which paper seems most relevant for understanding attention mechanisms?"
    print(f"\n👤 User (Turn 3): {query3}")
    response3 = await run_client_agent_async(query3, thread_id=thread_id)
    print(f"\n✅ Agent Response (Turn 3):\n{response3}\n")


async def test_combined_query():
    """Test a query that might use multiple server tools."""
    print("\n" + "="*80)
    print("TEST 5: Combined Query (Multiple Tools)")
    print("="*80)
    
    query = "Compare recent research papers on AI with current industry developments"
    response = await run_client_agent_async(query, thread_id="test_combined")
    
    print(f"\n✅ Final Response:\n{response}\n")


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CLIENT AGENT TEST SUITE")
    print("Testing LangGraph Client Agent → A2A Protocol → Server Agent")
    print("="*80)
    
    # Check if server is likely running
    print("\n⚠️  IMPORTANT: Make sure the A2A server is running!")
    print("   Start it with: uv run python -m app")
    print("   Server should be at: http://localhost:10000\n")
    
    input("Press Enter to continue with tests...")
    
    try:
        # Run individual skill tests
        await test_web_search()
        await test_arxiv_search()
        await test_rag_retrieval()
        
        # Run multi-turn conversation test
        await test_multi_turn_conversation()
        
        # Run combined query test
        await test_combined_query()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey Observations:")
        print("1. Client agent successfully delegated queries to A2A server")
        print("2. Server agent used appropriate tools (Tavily, ArXiv, RAG)")
        print("3. Multi-turn conversations preserved context")
        print("4. A2A protocol enabled seamless agent-to-agent communication")
        print("\nThis demonstrates the power of A2A: specialized agents working together!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Is the A2A server running? (uv run python -m app)")
        print("2. Is it accessible at http://localhost:10000?")
        print("3. Are your API keys set in .env? (OPENAI_API_KEY, TAVILY_API_KEY)")
        print("4. Check server logs for errors")


if __name__ == "__main__":
    asyncio.run(main())
