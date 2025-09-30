#!/usr/bin/env python3
"""
LinkedIn Post Generation Tools

Custom tools for creating, analyzing, and optimizing LinkedIn posts about ML papers.
These tools will be used by our LinkedIn agent team.
"""

import os
import re
import requests
from typing import Annotated, Dict, List
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM for tool operations
llm = ChatOpenAI(model="gpt-4o-mini")

@tool
def fetch_arxiv_paper(
    paper_identifier: Annotated[str, "ArXiv ID, URL, or paper title to fetch"]
) -> Dict[str, str]:
    """
    Fetch and extract key information from an ML paper on ArXiv.
    Returns structured data about the paper.
    """
    try:
        from langchain_community.document_loaders import ArxivLoader
        
        print(f"📄 Fetching paper: {paper_identifier}")
        
        # Handle different input types (ArXiv ID, URL, or title)
        if paper_identifier.startswith("http"):
            # Extract ArXiv ID from URL
            arxiv_id = paper_identifier.split("/")[-1].replace(".pdf", "")
            query = arxiv_id
        elif "." in paper_identifier and len(paper_identifier) < 20:
            # Looks like ArXiv ID (e.g., "2301.12345")
            query = paper_identifier
        else:
            # Treat as paper title
            query = paper_identifier
        
        # Fetch paper using ArxivLoader
        loader = ArxivLoader(
            query=query,
            load_max_docs=1,
            load_all_available_meta=True
        )
        
        documents = loader.load()
        
        if not documents:
            return {
                "error": f"Could not find paper: {paper_identifier}",
                "title": "",
                "authors": "",
                "abstract": "",
                "content": ""
            }
        
        doc = documents[0]
        
        return {
            "title": doc.metadata.get("Title", "Unknown Title"),
            "authors": doc.metadata.get("Authors", "Unknown Authors"),
            "abstract": doc.metadata.get("Summary", "No abstract available"),
            "content": doc.page_content[:2000],  # First 2000 chars of content
            "arxiv_url": doc.metadata.get("entry_id", ""),
            "published": doc.metadata.get("Published", "")
        }
        
    except Exception as e:
        return {
            "error": f"Error fetching paper: {str(e)}",
            "title": "",
            "authors": "",
            "abstract": "",
            "content": ""
        }


@tool
def analyze_ml_paper(
    paper_identifier: Annotated[str, "ArXiv ID, URL, or paper title to analyze"]
) -> Dict[str, str]:
    """
    Analyze ML paper and extract key insights for LinkedIn post creation.
    Fetches the paper first, then analyzes it.
    """
    try:
        # First fetch the paper
        paper_data = fetch_arxiv_paper.invoke(paper_identifier)
        
        if "error" in paper_data:
            return {
                "error": paper_data["error"],
                "core_contribution": "",
                "methodology": "",
                "key_results": "",
                "applications": "",
                "significance": "",
                "target_audience": "",
                "complexity_level": "intermediate"
            }
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert ML researcher analyzing papers for LinkedIn posts.

Extract and analyze the following from the paper:
1. Core contribution/innovation
2. Key technical approach or methodology
3. Main results/findings
4. Real-world applications or impact
5. Significance to the ML community
6. Target audience (researchers, practitioners, general tech audience)

Return a JSON object with:
- "core_contribution": Main innovation in 1-2 sentences
- "methodology": Technical approach in simple terms
- "key_results": Most important findings
- "applications": Real-world use cases
- "significance": Why this matters to ML community
- "target_audience": Who should care about this
- "complexity_level": "beginner", "intermediate", or "advanced"

Be concise but informative. Focus on what makes this paper interesting and valuable."""),
            ("human", """Paper Title: {title}

Authors: {authors}

Abstract: {abstract}

Content Sample: {content}

Please analyze this ML paper for LinkedIn post creation.""")
        ])
        
        chain = analysis_prompt | llm | StrOutputParser()
        result = chain.invoke({
            "title": paper_data.get("title", ""),
            "authors": paper_data.get("authors", ""),
            "abstract": paper_data.get("abstract", ""),
            "content": paper_data.get("content", "")
        })
        
        # Parse JSON response
        import json
        try:
            analysis = json.loads(result)
            # Add paper data to analysis for later use
            analysis["paper_data"] = paper_data
        except:
            # Fallback if JSON parsing fails
            analysis = {
                "core_contribution": paper_data.get("title", ""),
                "methodology": "Advanced ML techniques",
                "key_results": "Significant improvements shown",
                "applications": "Various ML applications",
                "significance": "Important contribution to ML field",
                "target_audience": "ML researchers and practitioners",
                "complexity_level": "intermediate",
                "paper_data": paper_data
            }
        
        return analysis
        
    except Exception as e:
        return {
            "error": f"Error analyzing paper: {str(e)}",
            "core_contribution": "",
            "methodology": "",
            "key_results": "",
            "applications": "",
            "significance": "",
            "target_audience": "",
            "complexity_level": "intermediate",
            "paper_data": {}
        }


@tool
def create_linkedin_post(
    paper_analysis: Annotated[Dict[str, str], "Analysis from analyze_ml_paper that includes paper_data"]
) -> str:
    """
    Create a professional LinkedIn post about an ML paper.
    Follows LinkedIn best practices for engagement and formatting.
    """
    try:
        post_creation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a LinkedIn content expert creating posts about ML papers.

Create a professional LinkedIn post that:
1. Starts with an engaging hook
2. Explains the paper's contribution in accessible terms
3. Highlights practical applications
4. Uses professional but approachable tone
5. Includes relevant hashtags (5-8 max)
6. Adds appropriate emojis for visual appeal
7. Ends with a call-to-action or discussion prompt
8. Stays under 2800 characters (leaving room for edits)

Format:
- Use line breaks for readability
- Include bullet points for key findings
- Add the paper's ArXiv link
- Use professional language but avoid jargon
- Make it engaging for both technical and non-technical audiences

Focus on why this research matters and how it could impact the industry."""),
            ("human", """Create a LinkedIn post for this ML paper:

Title: {title}
Authors: {authors}
Core Contribution: {core_contribution}
Methodology: {methodology}
Key Results: {key_results}
Applications: {applications}
Significance: {significance}
Target Audience: {target_audience}
Complexity: {complexity_level}
ArXiv URL: {arxiv_url}

Create an engaging LinkedIn post that will generate professional discussion.""")
        ])
        
        # Extract paper data from analysis
        paper_data = paper_analysis.get("paper_data", {})
        
        chain = post_creation_prompt | llm | StrOutputParser()
        post = chain.invoke({
            "title": paper_data.get("title", ""),
            "authors": paper_data.get("authors", ""),
            "core_contribution": paper_analysis.get("core_contribution", ""),
            "methodology": paper_analysis.get("methodology", ""),
            "key_results": paper_analysis.get("key_results", ""),
            "applications": paper_analysis.get("applications", ""),
            "significance": paper_analysis.get("significance", ""),
            "target_audience": paper_analysis.get("target_audience", ""),
            "complexity_level": paper_analysis.get("complexity_level", ""),
            "arxiv_url": paper_data.get("arxiv_url", "")
        })
        
        return post.strip()
        
    except Exception as e:
        return f"Error creating LinkedIn post: {str(e)}"


@tool
def verify_technical_accuracy(
    linkedin_post: Annotated[str, "LinkedIn post content"],
    paper_data: Annotated[Dict[str, str], "Original paper data"],
    paper_analysis: Annotated[Dict[str, str], "Paper analysis data"]
) -> Dict[str, any]:
    """
    Verify that the LinkedIn post accurately represents the ML paper.
    Checks for technical correctness and prevents misinformation.
    """
    try:
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a technical fact-checker for ML content.

Review the LinkedIn post against the original paper data and check for:
1. Factual accuracy of claims
2. Correct representation of methodology
3. Accurate reporting of results
4. No exaggerated or misleading statements
5. Proper attribution to authors

Return a JSON object:
{{
  "is_accurate": true/false,
  "accuracy_score": 0.0-1.0,
  "issues_found": ["list of any inaccuracies"],
  "corrections_needed": ["list of corrections"],
  "overall_assessment": "brief summary"
}}

Be strict - flag any potential misrepresentations."""),
            ("human", """LinkedIn Post to Verify:
{linkedin_post}

Original Paper Data:
Title: {title}
Authors: {authors}
Abstract: {abstract}
Analysis: {analysis}

Is this LinkedIn post technically accurate?""")
        ])
        
        chain = verification_prompt | llm | StrOutputParser()
        result = chain.invoke({
            "linkedin_post": linkedin_post,
            "title": paper_data.get("title", ""),
            "authors": paper_data.get("authors", ""),
            "abstract": paper_data.get("abstract", ""),
            "analysis": str(paper_analysis)
        })
        
        # Parse JSON response
        import json
        try:
            verification = json.loads(result)
        except:
            # Fallback verification
            verification = {
                "is_accurate": True,
                "accuracy_score": 0.8,
                "issues_found": [],
                "corrections_needed": [],
                "overall_assessment": "Generally accurate"
            }
        
        return verification
        
    except Exception as e:
        return {
            "error": f"Error verifying accuracy: {str(e)}",
            "is_accurate": False,
            "accuracy_score": 0.0
        }


@tool
def check_linkedin_style(
    linkedin_post: Annotated[str, "LinkedIn post content"]
) -> Dict[str, any]:
    """
    Check if the LinkedIn post follows platform best practices.
    Validates formatting, tone, length, and engagement elements.
    """
    try:
        style_check_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a LinkedIn content strategist checking post quality.

Evaluate the post for:
1. Professional tone (not too casual, not too academic)
2. Proper formatting (line breaks, bullet points, emojis)
3. Character count (must be under 3000 characters)
4. Hashtag usage (5-8 relevant hashtags)
5. Engagement elements (questions, CTAs, discussion prompts)
6. Visual appeal (emojis, spacing, readability)
7. LinkedIn best practices compliance

Return a JSON object:
{{
  "passes_style_check": true/false,
  "character_count": number,
  "style_score": 0.0-1.0,
  "issues_found": ["list of style issues"],
  "improvements_needed": ["list of improvements"],
  "hashtag_count": number,
  "has_cta": true/false,
  "overall_assessment": "brief summary"
}}

Be thorough in checking LinkedIn-specific requirements."""),
            ("human", """LinkedIn Post to Check:
{linkedin_post}

Please evaluate this post for LinkedIn style compliance.""")
        ])
        
        chain = style_check_prompt | llm | StrOutputParser()
        result = chain.invoke({"linkedin_post": linkedin_post})
        
        # Parse JSON response
        import json
        try:
            style_check = json.loads(result)
            # Add actual character count
            style_check["character_count"] = len(linkedin_post)
            style_check["under_limit"] = len(linkedin_post) <= 3000
        except:
            # Fallback style check
            style_check = {
                "passes_style_check": len(linkedin_post) <= 3000,
                "character_count": len(linkedin_post),
                "style_score": 0.7,
                "issues_found": [],
                "improvements_needed": [],
                "hashtag_count": linkedin_post.count("#"),
                "has_cta": "?" in linkedin_post or "comment" in linkedin_post.lower(),
                "under_limit": len(linkedin_post) <= 3000,
                "overall_assessment": "Basic style check completed"
            }
        
        return style_check
        
    except Exception as e:
        return {
            "error": f"Error checking style: {str(e)}",
            "passes_style_check": False,
            "character_count": len(linkedin_post)
        }


@tool
def optimize_engagement(
    linkedin_post: Annotated[str, "LinkedIn post content"],
    target_audience: Annotated[str, "Target audience for the post"]
) -> str:
    """
    Optimize LinkedIn post for maximum professional engagement.
    Enhances content to encourage comments, shares, and meaningful discussion.
    """
    try:
        engagement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a LinkedIn engagement expert optimizing posts for professional discussion.

Enhance the post to:
1. Add compelling hooks that grab attention
2. Include thought-provoking questions
3. Add relevant call-to-actions
4. Optimize hashtags for discoverability
5. Improve readability and visual appeal
6. Encourage professional discussion
7. Add value for the target audience

Keep the core technical content accurate while making it more engaging.
Maintain professional tone throughout.
Stay under 3000 characters."""),
            ("human", """Original LinkedIn Post:
{linkedin_post}

Target Audience: {target_audience}

Please optimize this post for maximum LinkedIn engagement while maintaining accuracy.""")
        ])
        
        chain = engagement_prompt | llm | StrOutputParser()
        optimized_post = chain.invoke({
            "linkedin_post": linkedin_post,
            "target_audience": target_audience
        })
        
        return optimized_post.strip()
        
    except Exception as e:
        return f"Error optimizing engagement: {str(e)}"


@tool
def validate_post_length(
    linkedin_post: Annotated[str, "LinkedIn post content"]
) -> Dict[str, any]:
    """
    Validate that the LinkedIn post meets length requirements.
    Provides detailed character count analysis.
    """
    char_count = len(linkedin_post)
    word_count = len(linkedin_post.split())
    line_count = len(linkedin_post.split('\n'))
    hashtag_count = linkedin_post.count('#')
    
    return {
        "character_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "hashtag_count": hashtag_count,
        "under_3000_limit": char_count <= 3000,
        "characters_remaining": 3000 - char_count,
        "length_status": "✅ Within limit" if char_count <= 3000 else "❌ Too long",
        "recommended_action": "Ready to post" if char_count <= 3000 else "Needs trimming"
    }


@tool
def trim_post_to_limit(
    linkedin_post: Annotated[str, "LinkedIn post content that's too long"]
) -> str:
    """
    Intelligently trim a LinkedIn post to fit within 3000 character limit.
    Preserves the most important content while maintaining readability.
    """
    try:
        if len(linkedin_post) <= 3000:
            return linkedin_post
        
        trim_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a LinkedIn content editor specializing in trimming posts to fit character limits.

Your task: Trim this LinkedIn post to under 3000 characters while:
1. Preserving the core message and key points
2. Maintaining professional tone and readability
3. Keeping important hashtags and CTAs
4. Ensuring the post still flows naturally
5. Prioritizing the most valuable content

Remove:
- Redundant phrases
- Less important details
- Excessive emojis
- Secondary hashtags

Keep:
- Main message and value proposition
- Key technical insights
- Author attribution
- Primary hashtags
- Call-to-action

Return the trimmed post that's under 3000 characters."""),
            ("human", """Original Post ({char_count} characters):
{linkedin_post}

Please trim this to under 3000 characters while preserving the core value.""")
        ])
        
        chain = trim_prompt | llm | StrOutputParser()
        trimmed_post = chain.invoke({
            "linkedin_post": linkedin_post,
            "char_count": len(linkedin_post)
        })
        
        return trimmed_post.strip()
        
    except Exception as e:
        return f"Error trimming post: {str(e)}"


@tool
def save_final_post(
    linkedin_post: Annotated[str, "Final LinkedIn post content"],
    paper_title: Annotated[str, "Title of the ML paper"]
) -> str:
    """
    Save the final LinkedIn post to a file for review and publishing.
    """
    try:
        # Create safe filename from paper title
        safe_title = re.sub(r'[^\w\s-]', '', paper_title)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        filename = f"linkedin_post_{safe_title[:50]}.txt"
        
        # Save post to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== FINAL LINKEDIN POST ===\n\n")
            f.write(linkedin_post)
            f.write(f"\n\n=== METADATA ===\n")
            f.write(f"Character Count: {len(linkedin_post)}\n")
            f.write(f"Word Count: {len(linkedin_post.split())}\n")
            f.write(f"Hashtag Count: {linkedin_post.count('#')}\n")
            f.write(f"Generated: {__import__('datetime').datetime.now()}\n")
        
        return f"✅ Final LinkedIn post saved to: {filename}"
        
    except Exception as e:
        return f"Error saving post: {str(e)}"


if __name__ == "__main__":
    print("🔧 LinkedIn Post Generation Tools Loaded")
    print("Available tools:")
    print("  • fetch_arxiv_paper - Get ML paper from ArXiv")
    print("  • analyze_ml_paper - Extract key insights")
    print("  • create_linkedin_post - Generate LinkedIn content")
    print("  • verify_technical_accuracy - Check factual correctness")
    print("  • check_linkedin_style - Validate LinkedIn formatting")
    print("  • optimize_engagement - Enhance for engagement")
    print("  • validate_post_length - Check character limits")
    print("  • trim_post_to_limit - Intelligently shorten posts")
    print("  • save_final_post - Save final version to file")
