#!/usr/bin/env python3
"""
LinkedIn Post Generation States

State definitions for the LinkedIn ML paper post generation system.
Following the exact patterns from the notebook's Task 4 and Task 5.
"""

import operator
from typing import Annotated, Dict, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

# Content Team State (following DocWritingState pattern)
class ContentTeamState(TypedDict):
    """
    State for the Content Research Team that fetches and analyzes ML papers
    and creates initial LinkedIn post drafts.
    
    Follows the exact pattern from DocWritingState in the notebook.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str  # Following notebook's file tracking pattern
    
    # LinkedIn-specific fields
    paper_input: str                    # ArXiv ID/URL/title from user
    paper_data: Dict[str, str]         # Raw paper data from fetch_arxiv_paper
    paper_analysis: Dict[str, str]     # Analysis from analyze_ml_paper
    draft_post: str                    # Initial LinkedIn post draft


# Verification Team State (following DocWritingState pattern)
class VerificationTeamState(TypedDict):
    """
    State for the Verification & Quality Team that checks technical accuracy,
    LinkedIn style compliance, and optimizes for engagement.
    
    Follows the exact pattern from DocWritingState in the notebook.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str  # Following notebook's file tracking pattern
    
    # LinkedIn verification-specific fields
    draft_post: str                         # Post to verify
    paper_data: Dict[str, str]             # Original paper for fact-checking
    paper_analysis: Dict[str, str]         # Analysis for verification
    verification_results: Dict[str, any]   # Technical accuracy results
    style_results: Dict[str, any]          # Style check results
    engagement_results: str                # Optimized version
    final_post: str                        # Final approved post


# Meta-Supervisor State (following Task 5's State pattern)
class LinkedInPostState(TypedDict):
    """
    Meta-supervisor state for orchestrating the entire LinkedIn post generation workflow.
    
    Follows the exact pattern from Task 5's State in the notebook.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    
    # LinkedIn workflow-specific fields
    paper_input: str    # Original user input (ArXiv ID/URL/title)
    final_post: str     # Final LinkedIn post output


# Helper function for file management (following notebook's prelude pattern)
def linkedin_prelude(state):
    """
    LinkedIn-specific prelude function for managing generated content files.
    Follows the exact pattern from the notebook's prelude function.
    """
    from pathlib import Path
    import os
    
    # Create LinkedIn content directory
    linkedin_dir = Path("./linkedin_content")
    if not linkedin_dir.exists():
        linkedin_dir.mkdir(exist_ok=True)
    
    # Track generated files
    generated_files = []
    try:
        generated_files = [
            f.relative_to(linkedin_dir) for f in linkedin_dir.rglob("*.txt")
        ]
    except:
        pass
    
    if not generated_files:
        return {**state, "current_files": "No LinkedIn posts generated yet."}
    
    return {
        **state,
        "current_files": "\nGenerated LinkedIn posts:\n"
        + "\n".join([f" - {f}" for f in generated_files]),
    }


# State initialization helpers (following notebook patterns)
def init_content_team_state(paper_input: str) -> ContentTeamState:
    """Initialize ContentTeamState with user input"""
    return {
        "messages": [],
        "team_members": "MLResearcher, LinkedInWriter",
        "next": "",
        "current_files": "",
        "paper_input": paper_input,
        "paper_data": {},
        "paper_analysis": {},
        "draft_post": ""
    }


def init_verification_team_state(draft_post: str, paper_data: Dict, paper_analysis: Dict) -> VerificationTeamState:
    """Initialize VerificationTeamState with content team output"""
    return {
        "messages": [],
        "team_members": "FactChecker, StyleChecker, EngagementOptimizer",
        "next": "",
        "current_files": "",
        "draft_post": draft_post,
        "paper_data": paper_data,
        "paper_analysis": paper_analysis,
        "verification_results": {},
        "style_results": {},
        "engagement_results": "",
        "final_post": ""
    }


def init_meta_state(paper_input: str) -> LinkedInPostState:
    """Initialize meta-supervisor state"""
    return {
        "messages": [],
        "next": "",
        "paper_input": paper_input,
        "final_post": ""
    }


if __name__ == "__main__":
    print("📊 LinkedIn Post Generation States Loaded")
    print("Available states:")
    print("  • ContentTeamState - For ML paper research and initial post creation")
    print("  • VerificationTeamState - For accuracy, style, and engagement verification")
    print("  • LinkedInPostState - For meta-supervisor orchestration")
    print("  • linkedin_prelude - File management helper")
    print("  • State initialization helpers included")
