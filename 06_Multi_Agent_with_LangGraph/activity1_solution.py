#!/usr/bin/env python3
"""
Activity #1 Solution: Dynamic ArXiv Paper Fetching for RAG System

This solution extends the original RAG implementation to dynamically fetch
ArXiv papers instead of using hard-coded documents.

Uses LangChain's ArxivLoader to fetch full PDF content, matching the notebook's approach.
The ArxivLoader downloads and processes full papers without requiring local file storage.
"""

import os
import getpass
import tiktoken
from typing import List, Annotated

# LangChain imports
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.tools import tool

# LangGraph imports
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict


def tiktoken_len(text: str) -> int:
    """Calculate token length using tiktoken"""
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)


# Initialize global components (matching notebook pattern)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
generator_llm = ChatOpenAI(model="gpt-4o-mini")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=0,
    length_function=tiktoken_len,
)

# Prompt template (matching notebook)
HUMAN_TEMPLATE = """
#CONTEXT:
{context}

QUERY:
{query}

Use the provided context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context respond with "I don't know"
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("human", HUMAN_TEMPLATE)
])


# Create custom tool for retrieving ArXiv papers (matching notebook's @tool pattern)
@tool
def retrieve_arxiv_information(
    query: Annotated[str, "query to search ArXiv papers for"]
) -> dict:
    """
    Use ArxivLoader to fetch and process full ArXiv papers dynamically.
    This matches the notebook's approach of using full PDF content with RAG.
    ArxivLoader downloads PDFs in memory without requiring local storage.
    """
    try:
        print(f"\n🔍 Searching ArXiv for papers on: {query}")
        
        # Use ArxivLoader to fetch full papers (like the notebook uses PyMuPDFLoader)
        # ArxivLoader handles PDF download and parsing automatically
        arxiv_loader = ArxivLoader(
            query=query,
            load_max_docs=2,  # Load top 2 most relevant papers
            load_all_available_meta=True
        )
        
        print("📥 Loading full papers from ArXiv (this may take a moment)...")
        documents = arxiv_loader.load()
        
        if not documents:
            print("❌ No relevant papers found on ArXiv for this query.")
            return {
                "context": [],
                "response": "No relevant papers found on ArXiv for this query."
            }
        
        print(f"✅ Loaded {len(documents)} papers:")
        for i, doc in enumerate(documents):
            title = doc.metadata.get('Title', 'Unknown Title')
            authors = doc.metadata.get('Authors', 'Unknown Authors')
            print(f"   {i+1}. {title}")
            print(f"      Authors: {authors}")
        
        # Split documents into chunks (matching notebook's approach)
        print(f"\n📄 Splitting documents into chunks...")
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks from the papers")
        
        # Create vector store (matching notebook's approach)
        print("🗄️  Creating vector store...")
        vectorstore = Qdrant.from_documents(
            documents=chunks,
            embedding=embedding_model,
            location=":memory:"
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever()
        
        # Retrieve relevant chunks for the query
        print("🔎 Retrieving relevant chunks...")
        retrieved_docs = retriever.invoke(query)
        print(f"✅ Retrieved {len(retrieved_docs)} relevant chunks")
        
        return {
            "context": retrieved_docs,
            "response": ""  # Will be generated later
        }
        
    except Exception as e:
        print(f"❌ Error processing ArXiv papers: {e}")
        return {
            "context": [],
            "response": f"Error retrieving information from ArXiv: {str(e)}"
        }


# LangGraph State (matching notebook pattern)
class State(TypedDict):
    question: str
    context: List[Document]
    response: str


def retrieve(state: State) -> dict:
    """
    Retrieve relevant documents from ArXiv papers.
    This replaces the hard-coded PDF retrieval with dynamic ArXiv fetching.
    """
    result = retrieve_arxiv_information.invoke(state["question"])
    return {"context": result["context"]}


def generate(state: State) -> dict:
    """Generate response using the retrieved context (matching notebook)"""
    if not state["context"]:
        return {"response": "I don't know - no relevant context was found in the ArXiv papers."}
    
    generator_chain = chat_prompt | generator_llm | StrOutputParser()
    response = generator_chain.invoke({
        "query": state["question"], 
        "context": state["context"]
    })
    return {"response": response}


def build_dynamic_arxiv_rag_graph():
    """Build the RAG graph (matching notebook's pattern exactly)"""
    rag_graph = StateGraph(State).add_sequence([retrieve, generate])
    rag_graph.add_edge(START, "retrieve")
    return rag_graph.compile()


def main():
    """Main function to demonstrate the dynamic ArXiv RAG system"""
    print("=" * 70)
    print("  Dynamic ArXiv RAG System")
    print("  (Matching the notebook's approach with full PDF content)")
    print("=" * 70)
    print("\nThis system dynamically fetches ArXiv papers and uses full PDF content")
    print("for RAG, just like the notebook but without hard-coding documents.\n")
    
    # Get API key
    if "OPENAI_API_KEY" not in os.environ:
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Build the compiled RAG graph
    print("🔧 Initializing Dynamic ArXiv RAG system...")
    compiled_rag_graph = build_dynamic_arxiv_rag_graph()
    print("✅ System ready!\n")
    
    # Example queries
    examples = [
        "What are the main challenges in multi-agent systems?",
        "How do large language models handle context windows?",
        "What are recent advances in retrieval augmented generation?"
    ]
    
    print("=" * 70)
    print("  Example Queries")
    print("=" * 70)
    
    for i, question in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"Example {i}: {question}")
        print('='*70)
        
        try:
            result = compiled_rag_graph.invoke({"question": question})
            print(f"\n📝 Response:\n{result['response']}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("  Interactive Mode")
    print("=" * 70)
    print("Enter your questions about any topic (or 'quit' to exit)")
    print("The system will search ArXiv and answer based on recent papers.\n")
    
    while True:
        question = input("\n💬 Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print()  # Add spacing
        try:
            result = compiled_rag_graph.invoke({"question": question})
            print(f"\n📝 Response:\n{result['response']}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()
