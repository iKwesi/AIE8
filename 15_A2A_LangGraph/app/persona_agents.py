"""CrewAI Persona-Based Agents that use the A2A server.

This module demonstrates using a different agent framework (CrewAI) to interact
with the A2A server. Each persona has unique goals and questioning styles.
"""
import asyncio
from typing import List, Dict, Any
from crewai import Agent, Task, Crew, Process
from app.crewai_a2a_tool import crewai_call_a2a_agent


def create_persona_agents() -> List[Agent]:
    """Create 4 persona agents with distinct personalities and goals.
    
    Returns:
        List of CrewAI Agent objects with human names and personalities
    """
    
    # Persona 1: Skeptical ML Researcher
    dr_elena_kovacs = Agent(
        role="Dr. Elena Kovács (Skeptical ML Researcher)",
        goal="Obtain detailed technical information with verifiable sources, benchmarks, and rigorous evidence",
        backstory="""You are Dr. Elena Kovács, a tenured professor at Stanford with a PhD 
        in Machine Learning from MIT. You've published over 60 papers in top-tier conferences 
        (NeurIPS, ICML, ICLR) and you're known for your rigorous peer reviews. You don't accept 
        claims without empirical evidence, benchmarks, and reproducible results. Surface-level 
        marketing speak frustrates you - you want technical depth, architectural details, and 
        comparative analysis. You always ask for sources and won't be satisfied until you have 
        concrete evidence to verify claims.""",
        tools=[crewai_call_a2a_agent],
        verbose=True,
        allow_delegation=False
    )
    
    # Persona 2: Curious AI Student
    marcus_chen = Agent(
        role="Marcus Chen (Curious AI Student)",
        goal="Understand complex AI concepts in simple, accessible terms with clear examples",
        backstory="""You are Marcus Chen, a bright computer science undergraduate at UC Berkeley 
        who's passionate about AI but still learning the fundamentals. You're smart and eager, 
        but you need explanations without heavy jargon. When you encounter technical terms, you 
        ask for clarification. You love analogies and real-world examples that make abstract 
        concepts concrete. You're not afraid to say "I don't understand" and you ask follow-up 
        questions until things click. Your goal is to truly grasp concepts, not just memorize them.""",
        tools=[crewai_call_a2a_agent],
        verbose=True,
        allow_delegation=False
    )
    
    # Persona 3: Senior AI Architect
    dr_priya_sharma = Agent(
        role="Dr. Priya Sharma (Senior AI Architect)",
        goal="Analyze system architecture, scalability, and implementation trade-offs",
        backstory="""You are Dr. Priya Sharma, Principal AI Architect at a Fortune 500 tech 
        company with 15 years of experience building production ML systems. You've architected 
        systems serving billions of users and you care deeply about scalability, reliability, 
        and operational excellence. You want to know: How does it scale? What are the latency 
        characteristics? What are the infrastructure requirements? What are the failure modes? 
        You think in terms of system design, not just algorithms. You need to understand the 
        engineering trade-offs and production considerations.""",
        tools=[crewai_call_a2a_agent],
        verbose=True,
        allow_delegation=False
    )
    
    # Persona 4: AI Business Strategist
    james_okonkwo = Agent(
        role="James Mensah (AI Business Strategist)",
        goal="Evaluate practical business applications, ROI, and strategic value",
        backstory="""You are James Mensah, VP of AI Strategy at a global consulting firm. 
        You help Fortune 500 companies adopt AI solutions and you've led over 50 successful 
        AI transformation projects. You don't care about technical details for their own sake - 
        you care about business impact. Your questions are: What problem does this solve? 
        What's the ROI? What are the use cases? How does it compare to alternatives? What's 
        the total cost of ownership? You need to justify AI investments to C-suite executives, 
        so you focus on measurable business value, competitive advantage, and practical 
        implementation timelines.""",
        tools=[crewai_call_a2a_agent],
        verbose=True,
        allow_delegation=False
    )
    
    return [dr_elena_kovacs, marcus_chen, dr_priya_sharma, james_okonkwo]


def create_persona_tasks(agents: List[Agent], topic: str) -> List[Task]:
    """Create tasks for each persona agent based on their unique perspectives.
    
    Args:
        agents: List of persona agents
        topic: The topic to research (e.g., "Kimi K2", "GPT-4", etc.)
        
    Returns:
        List of Task objects, one per persona
    """
    dr_elena, marcus, dr_priya, james = agents
    
    tasks = [
        Task(
            description=f"""Research {topic} from a rigorous academic perspective.
            
            Your approach:
            1. First, ask for technical papers, benchmarks, and documentation about {topic}
            2. Evaluate the response - does it have sources? Is it detailed enough?
            3. If not satisfied, ask for specific benchmarks, architectural details, or comparative analysis
            4. Continue until you have verifiable technical evidence or reach 3 attempts
            
            You must be thorough and demand evidence. Don't accept vague claims.""",
            agent=dr_elena,
            expected_output=f"Detailed technical analysis of {topic} with sources and benchmarks"
        ),
        
        Task(
            description=f"""Learn about {topic} in a way that's easy to understand.
            
            Your approach:
            1. First, ask for a simple explanation of what {topic} is and why it matters
            2. Evaluate the response - is it clear? Are there confusing terms?
            3. If not satisfied, ask for clarification, analogies, or simpler explanations
            4. Continue until you truly understand or reach 3 attempts
            
            Don't pretend to understand - ask follow-ups when things are unclear.""",
            agent=marcus,
            expected_output=f"Clear, accessible explanation of {topic} that a student can understand"
        ),
        
        Task(
            description=f"""Analyze {topic} from a system architecture perspective.
            
            Your approach:
            1. First, ask about the architecture, scalability, and implementation of {topic}
            2. Evaluate the response - does it cover system design? Performance characteristics?
            3. If not satisfied, ask about specific architectural decisions, trade-offs, or operational concerns
            4. Continue until you understand the engineering implications or reach 3 attempts
            
            Focus on production readiness and system-level considerations.""",
            agent=dr_priya,
            expected_output=f"Architectural analysis of {topic} with scalability and implementation insights"
        ),
        
        Task(
            description=f"""Evaluate {topic} from a business strategy perspective.
            
            Your approach:
            1. First, ask about practical applications, use cases, and business value of {topic}
            2. Evaluate the response - does it show ROI? Real-world applications?
            3. If not satisfied, ask about cost-benefit analysis, competitive advantages, or implementation timeline
            4. Continue until you can make a business case or reach 3 attempts
            
            Think like a C-suite executive - focus on measurable business impact.""",
            agent=james,
            expected_output=f"Business analysis of {topic} with ROI and strategic recommendations"
        )
    ]
    
    return tasks


async def run_persona_crew(topic: str) -> Dict[str, Any]:
    """Run all persona agents on a given topic.
    
    Args:
        topic: The topic to research (e.g., "Kimi K2")
        
    Returns:
        Dictionary with results from each persona
    """
    print(f"\n{'='*80}")
    print(f"🎭 PERSONA CREW ANALYSIS: {topic}")
    print(f"{'='*80}\n")
    
    # Create agents and tasks
    agents = create_persona_agents()
    tasks = create_persona_tasks(agents, topic)
    
    # Create crew with sequential process (one persona at a time)
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,  # Run personas one after another
        verbose=True
    )
    
    # Execute the crew
    print("Starting persona analysis...\n")
    result = crew.kickoff()
    
    return {
        "topic": topic,
        "result": result,
        "personas_executed": len(agents)
    }


def run_persona_crew_sync(topic: str) -> Dict[str, Any]:
    """Synchronous wrapper for run_persona_crew.
    
    Args:
        topic: The topic to research
        
    Returns:
        Dictionary with results from each persona
    """
    return asyncio.run(run_persona_crew(topic))
