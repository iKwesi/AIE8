#!/usr/bin/env python3
"""
LinkedIn System Testing

Test the complete LinkedIn ML paper post generation system.
Following the exact testing patterns from the notebook.
"""

import os
import getpass
from pathlib import Path

# Import our LinkedIn system components
from linkedin_content_team import content_chain, compiled_content_graph
from linkedin_verification_team import verification_chain, compiled_verification_graph
from linkedin_meta_supervisor import compiled_linkedin_system, create_linkedin_post_system
from linkedin_post_states import LinkedInPostState
from langchain_core.messages import HumanMessage

def setup_environment():
    """Setup API keys if not already set"""
    if "OPENAI_API_KEY" not in os.environ:
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key

def test_content_team():
    """Test Content Team - following notebook's authoring_chain.stream pattern"""
    print("🧪 Testing Content Team")
    print("=" * 50)
    
    # EXACT SAME PATTERN AS NOTEBOOK:
    for s in content_chain.stream(
        "Create a LinkedIn post about the ML paper: Attention Is All You Need",
        {"recursion_limit": 100},
    ):
        if "__end__" not in s:
            print(s)
            print("---")
    
    print("✅ Content team test completed!")

def test_verification_team():
    """Test Verification Team - following notebook's authoring_chain.stream pattern"""
    print("🧪 Testing Verification Team")
    print("=" * 50)
    
    # Test with sample LinkedIn post content
    sample_post = """🚀 Breakthrough in AI: "Attention Is All You Need" revolutionizes NLP!

Key innovations:
• Self-attention mechanisms replace recurrent layers
• Parallel processing dramatically speeds up training
• Transformer architecture becomes foundation for modern AI

Real-world impact:
✓ Powers ChatGPT, BERT, and modern language models
✓ Enables real-time translation and summarization
✓ Transforms how we build AI systems

This 2017 paper by Vaswani et al. literally changed everything we know about sequence modeling.

What's your experience with transformer models? How have they impacted your work?

#MachineLearning #AI #NLP #Transformers #DeepLearning #Research #Innovation

Paper: https://arxiv.org/abs/1706.03762"""
    
    # EXACT SAME PATTERN AS NOTEBOOK:
    for s in verification_chain.stream(
        f"Verify and optimize this LinkedIn post:\n\n{sample_post}",
        {"recursion_limit": 100},
    ):
        if "__end__" not in s:
            print(s)
            print("---")
    
    print("✅ Verification team test completed!")

def test_full_linkedin_system():
    """Test Full LinkedIn System - following notebook's compiled_super_graph.stream pattern"""
    print("🧪 Testing Full LinkedIn System")
    print("=" * 50)
    
    # Create linkedin content directory (like notebook's WORKING_DIRECTORY)
    linkedin_dir = Path("./linkedin_content")
    linkedin_dir.mkdir(exist_ok=True)
    
    # EXACT SAME PATTERN AS NOTEBOOK:
    for s in compiled_linkedin_system.stream(
        {
            "messages": [
                HumanMessage(
                    content="Create a professional LinkedIn post about the ML paper: Attention Is All You Need"
                )
            ],
            "paper_input": "Attention Is All You Need",
            "final_post": ""
        },
        {"recursion_limit": 30},
    ):
        if "__end__" not in s:
            print(s)
            print("---")
    
    print("✅ Full system test completed!")

def demo_different_paper_formats():
    """Demo the system with different paper input formats"""
    print("🎯 Demo: Different Paper Input Formats")
    print("=" * 50)
    
    test_papers = [
        "Attention Is All You Need",                    # Paper title
        "1706.03762",                                   # ArXiv ID
        "https://arxiv.org/abs/1706.03762",            # ArXiv URL
        "BERT: Pre-training of Deep Bidirectional Transformers",  # Another famous paper
    ]
    
    for paper in test_papers:
        print(f"\n📝 Testing with: {paper}")
        print("-" * 40)
        
        try:
            # Use the main system function
            create_linkedin_post_system(paper)
        except Exception as e:
            print(f"❌ Error with {paper}: {e}")
        
        print()

def main():
    """Main testing function"""
    print("🚀 LinkedIn ML Paper Post Generation System - Testing Suite")
    print("=" * 70)
    print("Following the exact testing patterns from the notebook!")
    print()
    
    # Setup environment
    setup_environment()
    
    # Run tests in order (like the notebook)
    print("1️⃣ Testing individual teams...")
    test_content_team()
    print("\n" + "="*70 + "\n")
    
    test_verification_team()
    print("\n" + "="*70 + "\n")
    
    print("2️⃣ Testing full integrated system...")
    test_full_linkedin_system()
    print("\n" + "="*70 + "\n")
    
    print("3️⃣ Demo with different paper formats...")
    demo_different_paper_formats()
    
    print("🎉 All tests completed!")
    print("\nGenerated files should be in ./linkedin_content/ directory")

if __name__ == "__main__":
    main()
