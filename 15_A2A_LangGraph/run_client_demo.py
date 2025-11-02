#!/usr/bin/env python3
"""Quick demo script for the client agent.

This script provides an interactive way to test the client agent
without running the full test suite.

Usage:
    python run_client_demo.py
"""
import asyncio
import sys
from dotenv import load_dotenv

from app.client_agent import run_client_agent_async


# Load environment variables
load_dotenv()


async def interactive_demo():
    """Run an interactive demo of the client agent."""
    print("\n" + "="*80)
    print("🤖 CLIENT AGENT INTERACTIVE DEMO")
    print("="*80)
    print("\nThis client agent uses the A2A protocol to communicate with the server agent.")
    print("The server has access to:")
    print("  • Web search (Tavily)")
    print("  • Academic papers (ArXiv)")
    print("  • Document retrieval (RAG)")
    print("\n" + "="*80)
    
    # Check server
    print("\n⚠️  IMPORTANT: Make sure the A2A server is running!")
    print("   Start it with: uv run python -m app")
    print("   Server should be at: http://localhost:10000\n")
    
    response = input("Is the server running? (y/n): ").strip().lower()
    if response != 'y':
        print("\n❌ Please start the server first, then run this script again.")
        sys.exit(0)
    
    print("\n" + "="*80)
    print("DEMO MODE")
    print("="*80)
    print("\nYou can ask questions and the client agent will delegate to the server.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Example queries
    print("Example queries you can try:")
    print("  1. What are the latest AI developments in 2025?")
    print("  2. Find papers on transformer architectures")
    print("  3. What do the documents say about AI usage?")
    print("  4. Compare recent AI research with industry trends\n")
    
    thread_id = "interactive_demo"
    turn = 1
    
    while True:
        try:
            # Get user input
            query = input(f"\n[Turn {turn}] Your query: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Run the client agent
            print(f"\n{'='*80}")
            print(f"Processing query (Turn {turn})...")
            print(f"{'='*80}\n")
            
            response = await run_client_agent_async(query, thread_id=thread_id)
            
            print(f"\n{'='*80}")
            print("✅ FINAL RESPONSE")
            print(f"{'='*80}")
            print(f"\n{response}\n")
            
            turn += 1
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nTroubleshooting:")
            print("1. Is the A2A server running?")
            print("2. Check your API keys in .env")
            print("3. Check server logs for errors\n")


async def quick_demo():
    """Run a quick non-interactive demo with predefined queries."""
    print("\n" + "="*80)
    print("🚀 CLIENT AGENT QUICK DEMO")
    print("="*80)
    print("\nRunning 3 quick test queries...\n")
    
    queries = [
        "What are the latest developments in AI?",
        "Find a paper on large language models",
        "What information is in the documents about AI usage?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(queries)}: {query}")
        print(f"{'='*80}\n")
        
        try:
            response = await run_client_agent_async(query, thread_id=f"quick_demo_{i}")
            print(f"\n✅ Response:\n{response}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
    
    print("\n" + "="*80)
    print("✅ QUICK DEMO COMPLETE")
    print("="*80)


async def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        await quick_demo()
    else:
        await interactive_demo()


if __name__ == "__main__":
    print("\n" + "🤖"*40)
    print("CLIENT AGENT DEMO")
    print("Demonstrating A2A Protocol: Client Agent → Server Agent")
    print("🤖"*40)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
