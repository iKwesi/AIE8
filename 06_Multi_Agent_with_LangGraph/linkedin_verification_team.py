#!/usr/bin/env python3
"""
LinkedIn Verification Team

Implementation of the Verification & Quality Team for LinkedIn ML paper posts.
Following the exact patterns from the notebook's Task 4 implementation.
"""

import functools
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# Import our custom components
from linkedin_post_tools import (
    verify_technical_accuracy,
    check_linkedin_style,
    validate_post_length,
    optimize_engagement,
    trim_post_to_limit,
    save_final_post
)
from linkedin_post_states import VerificationTeamState, linkedin_prelude

# Import notebook's helper functions (same as content team)
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


# Initialize LLM for verification team
verification_llm = ChatOpenAI(model="gpt-4o-mini")

# Create Verification Team Agents (following notebook's pattern)

# 1. Technical Fact Checker Agent
fact_checker_agent = create_agent(
    verification_llm,
    [verify_technical_accuracy],
    ("You are an expert technical fact checker who specializes in verifying the accuracy of ML content."
     "\nYour role is to:"
     "\n- Cross-check LinkedIn posts against original ML papers"
     "\n- Identify any technical inaccuracies or misrepresentations"
     "\n- Ensure proper attribution to authors"
     "\n- Flag exaggerated or misleading claims"
     "\n- Maintain strict standards for technical correctness"
     "\nBelow are files currently in your directory:\n{current_files}")
)

# Apply prelude for file context awareness (following notebook pattern)
context_aware_fact_checker = linkedin_prelude | fact_checker_agent
fact_checker_node = functools.partial(
    agent_node, agent=context_aware_fact_checker, name="FactChecker"
)

# 2. LinkedIn Style Checker Agent
style_checker_agent = create_agent(
    verification_llm,
    [check_linkedin_style, validate_post_length, trim_post_to_limit],
    ("You are an expert LinkedIn style reviewer who ensures posts meet platform standards."
     "\nYour role is to:"
     "\n- Check LinkedIn formatting, tone, and professional style"
     "\n- Enforce the 3000-character limit strictly"
     "\n- Validate hashtag usage (5-8 relevant hashtags)"
     "\n- Ensure proper use of emojis and visual elements"
     "\n- Trim posts intelligently when they exceed limits"
     "\n- Maintain professional but engaging tone"
     "\nBelow are files currently in your directory:\n{current_files}")
)

# Apply prelude for file context awareness (following notebook pattern)
context_aware_style_checker = linkedin_prelude | style_checker_agent
style_checker_node = functools.partial(
    agent_node, agent=context_aware_style_checker, name="StyleChecker"
)

# 3. Engagement Optimizer Agent
engagement_optimizer_agent = create_agent(
    verification_llm,
    [optimize_engagement, save_final_post],
    ("You are an expert LinkedIn engagement optimizer who maximizes post performance."
     "\nYour role is to:"
     "\n- Enhance posts for maximum professional engagement"
     "\n- Add compelling hooks and thought-provoking questions"
     "\n- Optimize call-to-actions for LinkedIn audience"
     "\n- Improve readability and visual appeal"
     "\n- Save final approved posts to files"
     "\n- Focus on generating meaningful professional discussion"
     "\nBelow are files currently in your directory:\n{current_files}")
)

# Apply prelude for file context awareness (following notebook pattern)
context_aware_engagement_optimizer = linkedin_prelude | engagement_optimizer_agent
engagement_optimizer_node = functools.partial(
    agent_node, agent=context_aware_engagement_optimizer, name="EngagementOptimizer"
)

# 4. Verification Team Supervisor (following notebook's authoring_supervisor_agent pattern)
verification_supervisor_agent = create_team_supervisor(
    verification_llm,
    ("You are a supervisor tasked with managing a conversation between the"
     " following workers: {team_members}. Your team specializes in verifying and optimizing"
     " LinkedIn posts about ML papers."
     "\nYou should always verify the technical contents and LinkedIn compliance after any edits are made."
     "\nGiven the following user request, respond with the worker to act next."
     " Each worker will perform a task and respond with their results and status."
     "\nWorkflow:"
     "\n1. First use FactChecker to verify technical accuracy"
     "\n2. Then use StyleChecker to ensure LinkedIn compliance and length limits"
     "\n3. Finally use EngagementOptimizer to enhance and save the final post"
     "\n4. When all verification tasks are complete, respond with FINISH."
     "\nWhen finished, respond with FINISH."),
    ["FactChecker", "StyleChecker", "EngagementOptimizer"]
)

# Build Verification Team Graph (following notebook's authoring_graph pattern)
verification_graph = StateGraph(VerificationTeamState)

# Add nodes
verification_graph.add_node("FactChecker", fact_checker_node)
verification_graph.add_node("StyleChecker", style_checker_node)
verification_graph.add_node("EngagementOptimizer", engagement_optimizer_node)
verification_graph.add_node("VerificationSupervisor", verification_supervisor_agent)

# Add edges (following notebook's pattern)
verification_graph.add_edge("FactChecker", "VerificationSupervisor")
verification_graph.add_edge("StyleChecker", "VerificationSupervisor")
verification_graph.add_edge("EngagementOptimizer", "VerificationSupervisor")

# Add conditional edges
verification_graph.add_conditional_edges(
    "VerificationSupervisor",
    lambda x: x["next"],
    {
        "FactChecker": "FactChecker",
        "StyleChecker": "StyleChecker",
        "EngagementOptimizer": "EngagementOptimizer",
        "FINISH": END,
    },
)

# Set entry point
verification_graph.set_entry_point("VerificationSupervisor")

# Compile the graph
compiled_verification_graph = verification_graph.compile()

# Create team chain interface (following notebook's pattern)
def enter_verification_chain(message: str, members: List[str]):
    """Interface for verification team - follows notebook's enter_authoring_chain pattern"""
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
    }
    return results

# Create the verification team chain
verification_chain = (
    functools.partial(enter_verification_chain, members=verification_graph.nodes)
    | compiled_verification_graph
)


def test_verification_team():
    """Test function to verify verification team works"""
    print("🧪 Testing Verification Team...")
    print("=" * 50)
    
    # Test with a sample LinkedIn post
    test_query = """Verify this LinkedIn post:
    
🚀 Exciting breakthrough in AI! New research shows transformers can achieve 95% accuracy on complex NLP tasks.

Key findings:
• Self-attention mechanisms revolutionize language understanding
• Parallel processing enables faster training
• Applications in translation, summarization, and more

This could transform how we build AI systems! What are your thoughts on the future of transformer architectures?

#MachineLearning #AI #NLP #Transformers #Research #Innovation #TechBreakthrough

Paper: https://arxiv.org/abs/1706.03762"""
    
    try:
        for step in verification_chain.stream(test_query, {"recursion_limit": 10}):
            if "__end__" not in step:
                print(step)
                print("---")
        print("✅ Verification team test completed!")
    except Exception as e:
        print(f"❌ Verification team test failed: {e}")


if __name__ == "__main__":
    print("🔍 LinkedIn Verification Team Loaded")
    print("=" * 50)
    print("Team Members:")
    print("  • FactChecker - Verifies technical accuracy against original paper")
    print("  • StyleChecker - Ensures LinkedIn compliance and character limits")
    print("  • EngagementOptimizer - Enhances for engagement and saves final post")
    print("  • VerificationSupervisor - Routes between team members")
    print()
    print("Available for testing:")
    print("  • compiled_verification_graph - The compiled team graph")
    print("  • verification_chain - The team chain interface")
    print("  • test_verification_team() - Test function")
    print()
    print("Usage:")
    print("  from linkedin_verification_team import verification_chain")
    print("  result = verification_chain.stream('Verify this post: [content]')")
