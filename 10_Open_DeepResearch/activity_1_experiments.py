"""
Activity 1: Configuration Experiments for Deep Research System

This script runs multiple experiments with different configurations to compare
their performance and quality. Uses a hybrid evaluation approach:
- Automatic quantitative metrics (execution time, tokens, sources, etc.)
- LLM-as-a-judge comparative evaluation (single API call)

Experiments:
1. Increased Parallelism (max_concurrent_research_units: 10)
2. Deeper Research (max_researcher_iterations: 8, max_react_tool_calls: 15)
3. Anthropic Native Search (search_api: "anthropic")
4. Disabled Clarification (allow_clarification: False)
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import os
import getpass
import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import uuid

# PDF processing
import PyPDF2

# LangChain and LangGraph imports
from langchain_anthropic import ChatAnthropic

# Import from open_deep_library
from open_deep_library.state import (
    AgentState,
    AgentInputState,
    SupervisorState,
    ResearcherState,
    ResearcherOutputState,
    ConductResearch,
    ResearchComplete,
    ClarifyWithUser,
    ResearchQuestion,
)

from open_deep_library.utils import (
    tavily_search,
    think_tool,
    get_all_tools,
    get_today_str,
)

from open_deep_library.configuration import (
    Configuration,
    SearchAPI,
)

from open_deep_library.prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    compress_research_system_prompt,
    final_report_generation_prompt,
)

from open_deep_library.deep_researcher import (
    clarify_with_user,
    write_research_brief,
    supervisor,
    supervisor_tools,
    researcher,
    researcher_tools,
    compress_research,
    final_report_generation,
    researcher_subgraph,
    supervisor_subgraph,
    deep_researcher,
)

# Set up API keys
print("Setting up API keys...")
if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")
if "TAVILY_API_KEY" not in os.environ:
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API key: ")

print("✓ API keys configured")

# ============================================================================
# SECTION 2: HELPER FUNCTIONS
# ============================================================================

def load_pdf(pdf_path: str) -> str:
    """Load and extract text from PDF."""
    pdf_text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() + "\n\n"
    return pdf_text


async def run_experiment(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    """
    Run a single experiment and collect metrics.
    
    Args:
        config: Configuration dictionary for the experiment
        experiment_name: Name of the experiment for logging
        
    Returns:
        Dictionary containing:
        - final_report: The generated research report
        - metrics: Quantitative metrics (time, tokens, sources, etc.)
        - config: The configuration used
    """
    print(f"\n{'='*60}")
    print(f"Starting: {experiment_name}")
    print(f"{'='*60}")
    
    # Track metrics
    start_time = time.time()
    metrics = {
        "experiment_name": experiment_name,
        "execution_time": 0,
        "num_sources": 0,
        "report_length": 0,
        "supervisor_iterations": 0,
        "researchers_spawned": 0,
    }
    
    # Run the research
    final_state = None
    async for event in deep_researcher.astream(
        {"messages": [{"role": "user", "content": research_request}]},
        config,
        stream_mode="updates"
    ):
        for node_name, node_output in event.items():
            # Track supervisor iterations
            if node_name == "supervisor" and "research_iterations" in node_output:
                metrics["supervisor_iterations"] = node_output["research_iterations"]
            
            # Track researchers spawned
            if node_name == "supervisor_tools" and "notes" in node_output:
                metrics["researchers_spawned"] = len(node_output["notes"])
            
            # Capture final state
            if node_name == "final_report_generation":
                final_state = node_output
    
    # Calculate final metrics
    end_time = time.time()
    metrics["execution_time"] = round(end_time - start_time, 2)
    
    if final_state and "final_report" in final_state:
        report = final_state["final_report"]
        metrics["report_length"] = len(report)
        # Count sources (rough estimate - count lines starting with numbers or bullets)
        metrics["num_sources"] = report.count("\n- ") + report.count("\n1. ") + report.count("\n2. ")
        
        print(f"\n✓ {experiment_name} completed in {metrics['execution_time']}s")
        print(f"  - Report length: {metrics['report_length']} characters")
        print(f"  - Sources found: {metrics['num_sources']}")
        print(f"  - Supervisor iterations: {metrics['supervisor_iterations']}")
        print(f"  - Researchers spawned: {metrics['researchers_spawned']}")
        
        return {
            "final_report": report,
            "metrics": metrics,
            "config": config
        }
    else:
        print(f"\n✗ {experiment_name} failed to generate report")
        return {
            "final_report": "ERROR: No report generated",
            "metrics": metrics,
            "config": config
        }


def display_metrics_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Display quantitative metrics in a formatted table."""
    print("\n" + "="*80)
    print("QUANTITATIVE METRICS COMPARISON")
    print("="*80)
    
    # Header
    print(f"\n{'Metric':<30} {'Exp 1':<12} {'Exp 2':<12} {'Exp 3':<12} {'Exp 4':<12}")
    print("-" * 80)
    
    # Execution time
    print(f"{'Execution Time (s)':<30} ", end="")
    for exp in ['exp1', 'exp2', 'exp3', 'exp4']:
        if exp in results:
            time_val = results[exp]['metrics']['execution_time']
            print(f"{time_val:<12.2f} ", end="")
    print()
    
    # Report length
    print(f"{'Report Length (chars)':<30} ", end="")
    for exp in ['exp1', 'exp2', 'exp3', 'exp4']:
        if exp in results:
            length = results[exp]['metrics']['report_length']
            print(f"{length:<12} ", end="")
    print()
    
    # Number of sources
    print(f"{'Number of Sources':<30} ", end="")
    for exp in ['exp1', 'exp2', 'exp3', 'exp4']:
        if exp in results:
            sources = results[exp]['metrics']['num_sources']
            print(f"{sources:<12} ", end="")
    print()
    
    # Supervisor iterations
    print(f"{'Supervisor Iterations':<30} ", end="")
    for exp in ['exp1', 'exp2', 'exp3', 'exp4']:
        if exp in results:
            iters = results[exp]['metrics']['supervisor_iterations']
            print(f"{iters:<12} ", end="")
    print()
    
    # Researchers spawned
    print(f"{'Researchers Spawned':<30} ", end="")
    for exp in ['exp1', 'exp2', 'exp3', 'exp4']:
        if exp in results:
            researchers = results[exp]['metrics']['researchers_spawned']
            print(f"{researchers:<12} ", end="")
    print()
    
    print("-" * 80)


async def evaluate_all_reports(results: Dict[str, Dict[str, Any]], baseline_report: str) -> str:
    """
    Use LLM to comparatively evaluate all reports.
    Single API call to rank all experiments.
    
    Args:
        results: Dictionary of experiment results
        baseline_report: The baseline report from the notebook
        
    Returns:
        String containing the comparative evaluation and rankings
    """
    print("\n" + "="*80)
    print("LLM COMPARATIVE EVALUATION")
    print("="*80)
    print("\nCalling Claude to evaluate and rank all reports...")
    
    # Prepare the evaluation prompt
    evaluation_prompt = f"""You are an expert research evaluator. You will compare 5 research reports and rank them from best to worst.

Evaluate based on these criteria:
1. Comprehensiveness - Does it cover all aspects of the research question?
2. Accuracy - Are findings well-supported and factually correct?
3. Clarity - Is it well-structured and easy to understand?
4. Depth - Does it provide meaningful insights beyond surface-level information?
5. Source Quality - Are sources credible and properly cited?

BASELINE REPORT (from original notebook):
{baseline_report[:3000]}...

EXPERIMENT 1 - Increased Parallelism (max_concurrent_research_units: 10):
{results['exp1']['final_report'][:3000]}...

EXPERIMENT 2 - Deeper Research (max_researcher_iterations: 8, max_react_tool_calls: 15):
{results['exp2']['final_report'][:3000]}...

EXPERIMENT 3 - Anthropic Native Search (search_api: "anthropic"):
{results['exp3']['final_report'][:3000]}...

EXPERIMENT 4 - Disabled Clarification (allow_clarification: False):
{results['exp4']['final_report'][:3000]}...

Please provide:
1. A ranking from 1st to 5th place
2. Brief justification for each ranking (2-3 sentences)
3. Overall winner and why
4. Key insights about which configuration works best for what scenarios

Format your response clearly with rankings and justifications."""

    # Call Claude for evaluation
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=2000)
    response = await llm.ainvoke(evaluation_prompt)
    
    return response.content


def display_rankings(evaluation: str) -> None:
    """Display the LLM evaluation results."""
    print("\n" + "="*80)
    print("COMPARATIVE RANKINGS")
    print("="*80)
    print(f"\n{evaluation}")
    print("\n" + "="*80)


# ============================================================================
# SECTION 3: LOAD PDF AND CREATE RESEARCH QUESTION
# ============================================================================

print("\nLoading PDF document...")
pdf_path = "data/howpeopleuseai.pdf"
pdf_content = load_pdf(pdf_path)
print(f"✓ Loaded PDF with {len(pdf_content)} characters")

# Create research request (same as notebook)
research_request = f"""
I have a PDF document about how people use AI. Please analyze this document and provide insights about:

1. What are the main findings about how people are using AI?
2. What are the most common use cases?
3. What trends or patterns emerge from the data?

Here's the PDF content:

{pdf_content[:10000]}  # First 10k chars to stay within limits

...[content truncated for context window]
"""

print("✓ Research question prepared")

# ============================================================================
# SECTION 4: BASELINE REFERENCE
# ============================================================================

# NOTE: Baseline experiment already run in main notebook with config:
# - max_concurrent_research_units: 1
# - max_researcher_iterations: 2  
# - max_react_tool_calls: 3
# - search_api: "tavily"
# - allow_clarification: True
#
# The baseline report will be loaded from the notebook output for comparison.
# All experiments below will be compared against those baseline results.

# For this script, we'll use a placeholder or load from file if available
baseline_report = """[PLACEHOLDER: Copy baseline report from notebook here, or load from file]

This should be the final report generated in the main notebook with the original configuration.
"""

print("\n" + "="*60)
print("BASELINE CONFIGURATION (from notebook)")
print("="*60)
print("- max_concurrent_research_units: 1")
print("- max_researcher_iterations: 2")
print("- max_react_tool_calls: 3")
print("- search_api: tavily")
print("- allow_clarification: True")
print("\nBaseline report will be used for comparison in LLM evaluation.")

# ============================================================================
# SECTION 5: EXPERIMENT 1 - INCREASED PARALLELISM
# ============================================================================

print("\n" + "="*60)
print("EXPERIMENT 1: INCREASED PARALLELISM")
print("="*60)
print("Configuration: max_concurrent_research_units = 10")
print("Hypothesis: More parallel researchers = faster execution, broader coverage")

config_exp1 = {
    "configurable": {
        # Model configuration
        "research_model": "anthropic:claude-sonnet-4-20250514",
        "research_model_max_tokens": 10000,
        "compression_model": "anthropic:claude-sonnet-4-20250514",
        "compression_model_max_tokens": 8192,
        "final_report_model": "anthropic:claude-sonnet-4-20250514",
        "final_report_model_max_tokens": 10000,
        "summarization_model": "anthropic:claude-sonnet-4-20250514",
        "summarization_model_max_tokens": 8192,
        
        # Research behavior - INCREASED PARALLELISM
        "allow_clarification": True,
        "max_concurrent_research_units": 10,  # ← CHANGED: 10 parallel researchers
        "max_researcher_iterations": 2,
        "max_react_tool_calls": 3,
        
        # Search configuration
        "search_api": "tavily",
        "max_content_length": 50000,
        
        # Thread ID
        "thread_id": str(uuid.uuid4())
    }
}

# ============================================================================
# SECTION 6: EXPERIMENT 2 - DEEPER RESEARCH
# ============================================================================

print("\n" + "="*60)
print("EXPERIMENT 2: DEEPER RESEARCH")
print("="*60)
print("Configuration: max_researcher_iterations = 8, max_react_tool_calls = 15")
print("Hypothesis: More iterations = deeper insights, more comprehensive coverage")

config_exp2 = {
    "configurable": {
        # Model configuration
        "research_model": "anthropic:claude-sonnet-4-20250514",
        "research_model_max_tokens": 10000,
        "compression_model": "anthropic:claude-sonnet-4-20250514",
        "compression_model_max_tokens": 8192,
        "final_report_model": "anthropic:claude-sonnet-4-20250514",
        "final_report_model_max_tokens": 10000,
        "summarization_model": "anthropic:claude-sonnet-4-20250514",
        "summarization_model_max_tokens": 8192,
        
        # Research behavior - DEEPER RESEARCH
        "allow_clarification": True,
        "max_concurrent_research_units": 1,
        "max_researcher_iterations": 8,  # ← CHANGED: More supervisor iterations
        "max_react_tool_calls": 15,      # ← CHANGED: More tool calls per researcher
        
        # Search configuration
        "search_api": "tavily",
        "max_content_length": 50000,
        
        # Thread ID
        "thread_id": str(uuid.uuid4())
    }
}

# ============================================================================
# SECTION 7: EXPERIMENT 3 - ANTHROPIC NATIVE SEARCH
# ============================================================================

print("\n" + "="*60)
print("EXPERIMENT 3: ANTHROPIC NATIVE SEARCH")
print("="*60)
print("Configuration: search_api = 'anthropic'")
print("Hypothesis: Native search integration may provide better quality results")

config_exp3 = {
    "configurable": {
        # Model configuration
        "research_model": "anthropic:claude-sonnet-4-20250514",
        "research_model_max_tokens": 10000,
        "compression_model": "anthropic:claude-sonnet-4-20250514",
        "compression_model_max_tokens": 8192,
        "final_report_model": "anthropic:claude-sonnet-4-20250514",
        "final_report_model_max_tokens": 10000,
        "summarization_model": "anthropic:claude-sonnet-4-20250514",
        "summarization_model_max_tokens": 8192,
        
        # Research behavior
        "allow_clarification": True,
        "max_concurrent_research_units": 1,
        "max_researcher_iterations": 2,
        "max_react_tool_calls": 3,
        
        # Search configuration - ANTHROPIC NATIVE SEARCH
        "search_api": "anthropic",  # ← CHANGED: Use Anthropic's native search
        "max_content_length": 50000,
        
        # Thread ID
        "thread_id": str(uuid.uuid4())
    }
}

# ============================================================================
# SECTION 8: EXPERIMENT 4 - DISABLED CLARIFICATION
# ============================================================================

print("\n" + "="*60)
print("EXPERIMENT 4: DISABLED CLARIFICATION")
print("="*60)
print("Configuration: allow_clarification = False")
print("Hypothesis: Skipping clarification may speed up workflow but reduce accuracy")

config_exp4 = {
    "configurable": {
        # Model configuration
        "research_model": "anthropic:claude-sonnet-4-20250514",
        "research_model_max_tokens": 10000,
        "compression_model": "anthropic:claude-sonnet-4-20250514",
        "compression_model_max_tokens": 8192,
        "final_report_model": "anthropic:claude-sonnet-4-20250514",
        "final_report_model_max_tokens": 10000,
        "summarization_model": "anthropic:claude-sonnet-4-20250514",
        "summarization_model_max_tokens": 8192,
        
        # Research behavior - DISABLED CLARIFICATION
        "allow_clarification": False,  # ← CHANGED: Skip clarification phase
        "max_concurrent_research_units": 1,
        "max_researcher_iterations": 2,
        "max_react_tool_calls": 3,
        
        # Search configuration
        "search_api": "tavily",
        "max_content_length": 50000,
        
        # Thread ID
        "thread_id": str(uuid.uuid4())
    }
}

# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

async def main():
    """
    Main execution function that:
    1. Runs each experiment once
    2. Collects results as they complete
    3. Displays metrics summary
    4. Runs LLM evaluation once
    5. Shows final rankings
    """
    
    print("\n" + "="*80)
    print("STARTING ACTIVITY 1 EXPERIMENTS")
    print("="*80)
    print("\nThis will run 4 experiments with different configurations.")
    print("Each experiment will be tracked for metrics and evaluated by LLM.")
    print("\nEstimated time: 5-10 minutes depending on configurations")
    print("="*80)
    
    # Dictionary to store all results
    results = {}
    
    # Run Experiment 1
    print("\n🔬 Running Experiment 1: Increased Parallelism...")
    results['exp1'] = await run_experiment(config_exp1, "Experiment 1: Increased Parallelism")
    
    # Run Experiment 2
    print("\n🔬 Running Experiment 2: Deeper Research...")
    results['exp2'] = await run_experiment(config_exp2, "Experiment 2: Deeper Research")
    
    # Run Experiment 3
    print("\n🔬 Running Experiment 3: Anthropic Native Search...")
    results['exp3'] = await run_experiment(config_exp3, "Experiment 3: Anthropic Native Search")
    
    # Run Experiment 4
    print("\n🔬 Running Experiment 4: Disabled Clarification...")
    results['exp4'] = await run_experiment(config_exp4, "Experiment 4: Disabled Clarification")
    
    # Display metrics summary
    display_metrics_table(results)
    
    # Run LLM comparative evaluation
    evaluation = await evaluate_all_reports(results, baseline_report)
    display_rankings(evaluation)
    
    # Save results to file for later reference
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results_to_save = {
        exp_name: {
            "metrics": exp_data["metrics"],
            "report_preview": exp_data["final_report"][:500] + "..."
        }
        for exp_name, exp_data in results.items()
    }
    
    with open("experiment_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2)
    
    print("✓ Results saved to experiment_results.json")
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Review the quantitative metrics to understand performance trade-offs")
    print("2. Check the LLM evaluation for qualitative insights")
    print("3. Consider which configuration best fits your use case")
    print("\nNext Steps:")
    print("- Copy sections to notebook for interactive exploration")
    print("- Try additional configuration combinations")
    print("- Analyze specific reports in detail")


if __name__ == "__main__":
    asyncio.run(main())
