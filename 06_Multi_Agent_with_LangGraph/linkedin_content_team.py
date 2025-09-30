#!/usr/bin/env python3
"""
LinkedIn Content Creation Team

Implementation of the Content Research Team for LinkedIn ML paper posts.
Following the exact patterns from the notebook's Task 4 implementation.
"""

import functools
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# Import our custom components
from linkedin_post_tools import (
    fetch_arxiv_paper, 
    analyze_ml_paper, 
    create_linkedin_post
)
from linkedin_post_states import ContentTeamState, linkedin_prelude

# Import notebook's helper functions (we'll need to define these)
def agent_node(state, agent, name):
    """Agent node helper - exact copy from notebook"""
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def create_agent(llm, tools, system_prompt):
    """Create agent helper - exact copy from notebook"""
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    system_prompt += ("\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason!")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


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


# Initialize LLM for content team
content_llm = ChatOpenAI(model="gpt-4o-mini")

# Create Content Team Agents (following notebook's pattern)

# 1. ML Paper Researcher Agent
ml_researcher_agent = create_agent(
    content_llm,
    [fetch_arxiv_paper, analyze_ml_paper],
    ("You are an expert ML researcher who specializes in fetching and analyzing machine learning papers from ArXiv."
     "\nYour role is to:"
     "\n- Fetch ML papers using ArXiv IDs, URLs, or titles"
     "\n- Extract key technical insights, methodology, and significance"
     "\n- Provide structured analysis for LinkedIn content creation"
     "\n- Focus on practical applications and real-world impact"
     "\nBelow are files currently in your directory:\n{current_files}")
)

# Apply prelude for file context awareness (following notebook pattern)
context_aware_ml_researcher = linkedin_prelude | ml_researcher_agent
ml_researcher_node = functools.partial(
    agent_node, agent=context_aware_ml_researcher, name="MLResearcher"
)

# 2. LinkedIn Content Writer Agent
linkedin_writer_agent = create_agent(
    content_llm,
    [create_linkedin_post],
    ("You are an expert LinkedIn content writer who specializes in creating engaging posts about machine learning research."
     "\nYour role is to:"
     "\n- Create professional LinkedIn posts from ML paper analysis"
     "\n- Use engaging hooks, clear explanations, and proper formatting"
     "\n- Include relevant hashtags and call-to-actions"
     "\n- Maintain professional tone while being accessible"
     "\n- Stay under 2800 characters to leave room for edits"
     "\nBelow are files currently in your directory:\n{current_files}")
)

# Apply prelude for file context awareness (following notebook pattern)
context_aware_linkedin_writer = linkedin_prelude | linkedin_writer_agent
linkedin_writer_node = functools.partial(
    agent_node, agent=context_aware_linkedin_writer, name="LinkedInWriter"
)

# 3. Content Team Supervisor (following notebook's pattern)
content_supervisor_agent = create_team_supervisor(
    content_llm,
    ("You are a supervisor tasked with managing a conversation between the"
     " following workers: {team_members}. Your team specializes in researching ML papers"
     " and creating LinkedIn content."
     "\nGiven the following user request, respond with the worker to act next."
     " Each worker will perform a task and respond with their results and status."
     "\nWorkflow:"
     "\n1. First use MLResearcher to fetch and analyze the ML paper"
     "\n2. Then use LinkedInWriter to create the LinkedIn post"
     "\n3. When both tasks are complete, respond with FINISH."
     "\nYou should only pass tasks to workers that are specifically content creation focused."
     " When finished, respond with FINISH."),
    ["MLResearcher", "LinkedInWriter"]
)

# Build Content Team Graph (following notebook's authoring_graph pattern)
content_graph = StateGraph(ContentTeamState)

# Add nodes
content_graph.add_node("MLResearcher", ml_researcher_node)
content_graph.add_node("LinkedInWriter", linkedin_writer_node)
content_graph.add_node("ContentSupervisor", content_supervisor_agent)

# Add edges (following notebook's pattern)
content_graph.add_edge("MLResearcher", "ContentSupervisor")
content_graph.add_edge("LinkedInWriter", "ContentSupervisor")

# Add conditional edges
content_graph.add_conditional_edges(
    "ContentSupervisor",
    lambda x: x["next"],
    {
        "MLResearcher": "MLResearcher",
        "LinkedInWriter": "LinkedInWriter",
        "FINISH": END,
    },
)

# Set entry point
content_graph.set_entry_point("ContentSupervisor")

# Compile the graph
compiled_content_graph = content_graph.compile()

# Create team chain interface (following notebook's pattern)
def enter_content_chain(message: str, members: List[str]):
    """Interface for content team - follows notebook's enter_authoring_chain pattern"""
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
    }
    return results

# Create the content team chain
content_chain = (
    functools.partial(enter_content_chain, members=content_graph.nodes)
    | compiled_content_graph
)


def test_content_team():
    """Test function to verify content team works"""
    print("🧪 Testing Content Team...")
    print("=" * 50)
    
    # Test with a sample ML paper
    test_query = "Create a LinkedIn post about the paper: Attention Is All You Need"
    
    try:
        for step in content_chain.stream(test_query, {"recursion_limit": 10}):
            if "__end__" not in step:
                print(step)
                print("---")
        print("✅ Content team test completed!")
    except Exception as e:
        print(f"❌ Content team test failed: {e}")


if __name__ == "__main__":
    print("👥 LinkedIn Content Creation Team Loaded")
    print("=" * 50)
    print("Team Members:")
    print("  • MLResearcher - Fetches and analyzes ML papers from ArXiv")
    print("  • LinkedInWriter - Creates professional LinkedIn posts")
    print("  • ContentSupervisor - Routes between team members")
    print()
    print("Available for testing:")
    print("  • compiled_content_graph - The compiled team graph")
    print("  • content_chain - The team chain interface")
    print("  • test_content_team() - Test function")
    print()
    print("Usage:")
    print("  from linkedin_content_team import content_chain")
    print("  result = content_chain.stream('Create post about paper: [ArXiv ID]')")
