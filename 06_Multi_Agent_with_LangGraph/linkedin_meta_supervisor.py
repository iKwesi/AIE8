#!/usr/bin/env python3
"""
LinkedIn Meta-Supervisor and Full Graph

Implementation of the Meta-Supervisor that orchestrates Content and Verification teams.
Following the exact patterns from the notebook's Task 5 implementation.
"""

import functools
import operator
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# Import our team components
from linkedin_content_team import content_chain
from linkedin_verification_team import verification_chain
from linkedin_post_states import LinkedInPostState

# Import notebook's helper functions
def create_team_supervisor(llm, system_prompt, members):
    """Create team supervisor - exact copy from notebook"""
    from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [{"enum": options}],
                },
            },
            "required": ["next"],
        },
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next?"
         " Or should we FINISH? Select one of: {options}"),
    ]).partial(options=str(options), team_members=", ".join(members))
    
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )


# Helper functions for team communication (following notebook's Task 5 pattern)
def get_last_message(state: LinkedInPostState) -> str:
    """Extract last message content - exact copy from notebook"""
    return state["messages"][-1].content


def join_graph(response: dict):
    """Join graph response - exact copy from notebook"""
    return {"messages": [response["messages"][-1]]}


# Initialize Meta-Supervisor LLM
meta_llm = ChatOpenAI(model="gpt-4o-mini")

# Create Meta-Supervisor Agent (following notebook's super_supervisor_agent pattern)
linkedin_meta_supervisor_agent = create_team_supervisor(
    meta_llm,
    ("You are a meta-supervisor tasked with managing a conversation between the"
     " following teams: {team_members}. You orchestrate the LinkedIn ML paper post generation workflow."
     "\nGiven the following user request, respond with the team to act next."
     " Each team will perform their specialized tasks and respond with their results and status."
     "\nWorkflow:"
     "\n1. First route to 'Content team' to research the ML paper and create initial LinkedIn post"
     "\n2. Then route to 'Verification team' to verify accuracy, style, and optimize engagement"
     "\n3. When all teams have finished their work, respond with FINISH."
     "\nWhen all workers are finished, you must respond with FINISH."),
    ["Content team", "Verification team"]
)

# Build Meta-Supervisor Graph (following notebook's super_graph pattern)
linkedin_meta_graph = StateGraph(LinkedInPostState)

# Add team nodes (following notebook's pattern with get_last_message | team_chain | join_graph)
linkedin_meta_graph.add_node(
    "Content team", 
    get_last_message | content_chain | join_graph
)
linkedin_meta_graph.add_node(
    "Verification team", 
    get_last_message | verification_chain | join_graph
)
linkedin_meta_graph.add_node("LinkedInMetaSupervisor", linkedin_meta_supervisor_agent)

# Add edges (following notebook's super_graph pattern)
linkedin_meta_graph.add_edge("Content team", "LinkedInMetaSupervisor")
linkedin_meta_graph.add_edge("Verification team", "LinkedInMetaSupervisor")

# Add conditional edges
linkedin_meta_graph.add_conditional_edges(
    "LinkedInMetaSupervisor",
    lambda x: x["next"],
    {
        "Content team": "Content team",
        "Verification team": "Verification team",
        "FINISH": END,
    },
)

# Set entry point
linkedin_meta_graph.set_entry_point("LinkedInMetaSupervisor")

# Compile the full LinkedIn system
compiled_linkedin_system = linkedin_meta_graph.compile()


def create_linkedin_post_system(paper_input: str):
    """
    Main function to generate LinkedIn post about ML paper.
    Following the notebook's usage pattern.
    """
    print("🚀 LinkedIn ML Paper Post Generation System")
    print("=" * 60)
    print(f"📄 Processing paper: {paper_input}")
    print()
    
    try:
        # Initialize the system with user input
        initial_state = {
            "messages": [HumanMessage(content=f"Create a professional LinkedIn post about this ML paper: {paper_input}")],
            "paper_input": paper_input,
            "final_post": ""
        }
        
        # Run the full system
        print("🔄 Starting LinkedIn post generation workflow...")
        for step in compiled_linkedin_system.stream(initial_state, {"recursion_limit": 20}):
            if "__end__" not in step:
                print(step)
                print("---")
        
        print("✅ LinkedIn post generation completed!")
        
    except Exception as e:
        print(f"❌ Error in LinkedIn system: {e}")


def test_full_system():
    """Test the complete LinkedIn post generation system"""
    print("🧪 Testing Full LinkedIn System...")
    print("=" * 60)
    
    # Test with famous ML papers
    test_papers = [
        "Attention Is All You Need",
        "2017.11499",  # ArXiv ID format
        "https://arxiv.org/abs/1706.03762"  # URL format
    ]
    
    for paper in test_papers:
        print(f"\n📝 Testing with: {paper}")
        print("-" * 40)
        create_linkedin_post_system(paper)
        print()


if __name__ == "__main__":
    print("🎯 LinkedIn Meta-Supervisor and Full System Loaded")
    print("=" * 60)
    print("System Architecture:")
    print("  🎯 LinkedInMetaSupervisor - Orchestrates the entire workflow")
    print("  👥 Content team - Researches papers and creates initial posts")
    print("  🔍 Verification team - Verifies accuracy, style, and optimizes engagement")
    print()
    print("Available functions:")
    print("  • compiled_linkedin_system - The complete multi-agent system")
    print("  • create_linkedin_post_system(paper_input) - Main generation function")
    print("  • test_full_system() - Test with multiple paper formats")
    print()
    print("Usage:")
    print("  from linkedin_meta_supervisor import create_linkedin_post_system")
    print("  create_linkedin_post_system('Attention Is All You Need')")
    print("  create_linkedin_post_system('2301.12345')  # ArXiv ID")
    print("  create_linkedin_post_system('https://arxiv.org/abs/1706.03762')  # URL")
