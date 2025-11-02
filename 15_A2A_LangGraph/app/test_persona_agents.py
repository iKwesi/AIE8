"""Test script for CrewAI persona-based agents.

This script demonstrates 4 different personas (with human names) using the A2A server
to research a topic. Each persona has unique goals and questioning styles.

Prerequisites:
- The A2A server must be running: make server (or uv run python -m app)
- Server should be available at http://localhost:10000
"""
import asyncio
from dotenv import load_dotenv
from app.persona_agents import run_persona_crew


# Load environment variables
load_dotenv()


async def main():
    """Run persona crew analysis on a topic."""
    print("\n" + "="*40)
    print("🤖 CREWAI PERSONA-BASED AGENT DEMONSTRATION")
    print("Testing 4 Different Personas Using A2A Protocol")
    print("="*40)
    
    # Check server
    print("\n⚠️  IMPORTANT: Make sure the A2A server is running!")
    print("   Start it with: make server")
    print("   Or: uv run python -m app")
    print("   Server should be at: http://localhost:10000\n")
    
    response = input("Is the server running? (y/n): ").strip().lower()
    if response != 'y':
        print("\n❌ Please start the server first, then run this script again.")
        return
    
    print("\n" + "="*80)
    print("PERSONA LINEUP")
    print("="*80)
    print("\n1. 🔬 Dr. Elena Kovács - Skeptical ML Researcher")
    print("   Goal: Demands technical papers, benchmarks, and rigorous evidence")
    print("   Style: Won't accept claims without sources\n")
    
    print("2. 🎓 Marcus Chen - Curious AI Student")
    print("   Goal: Wants simple explanations with clear examples")
    print("   Style: Asks follow-ups when things aren't clear\n")
    
    print("3. 💻 Dr. Priya Sharma - Senior AI Architect")
    print("   Goal: Analyzes architecture, scalability, and trade-offs")
    print("   Style: Thinks in terms of production systems\n")
    
    print("4. 💼 James Mensah - AI Business Strategist")
    print("   Goal: Evaluates ROI and business value")
    print("   Style: Focuses on measurable business impact\n")
    
    # Get topic from user
    print("="*80)
    print("TOPIC SELECTION")
    print("="*80)
    print("\nSuggested topics:")
    print("  • Kimi K2")
    print("  • GPT-4")
    print("  • Claude 3.5 Sonnet")
    print("  • Llama 3")
    print("  • Gemini Pro")
    print("  • Or any AI model/technology you're curious about\n")
    
    topic = input("Enter a topic to research: ").strip()
    if not topic:
        topic = "Kimi K2"  # Default topic
        print(f"Using default topic: {topic}")
    
    print(f"\n{'='*80}")
    print(f"STARTING ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTopic: {topic}")
    print("Each persona will research this topic from their unique perspective...")
    print("This may take a few minutes as each persona may ask multiple questions.\n")
    
    input("Press Enter to begin...")
    
    try:
        # Run the persona crew
        result = await run_persona_crew(topic)
        
        print(f"\n{'='*80}")
        print("✅ ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"\nTopic: {result['topic']}")
        print(f"Personas Executed: {result['personas_executed']}")
        print("\nFinal Result:")
        print(result['result'])
        
        print("\n" + "="*80)
        print("KEY OBSERVATIONS")
        print("="*80)
        print("\n1. Each persona asked questions aligned with their goals")
        print("2. The A2A server handled diverse query types (technical, simple, business)")
        print("3. Personas demonstrated autonomous follow-up behavior")
        print("4. CrewAI successfully orchestrated multiple agents using A2A protocol")
        print("\nThis demonstrates how different agent frameworks can interoperate via A2A! 🎉")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Is the A2A server running? (make server)")
        print("2. Is it accessible at http://localhost:10000?")
        print("3. Are your API keys set in .env? (OPENAI_API_KEY, TAVILY_API_KEY)")
        print("4. Check server logs for errors")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 CREWAI PERSONA AGENTS")
    print("Demonstrating A2A Protocol with Different Agent Framework")
    print("="*50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
