# Retriever Evaluation Pipeline
# This file contains all sections for comprehensive retriever evaluation
# Copy each section into separate notebook cells as needed

# =============================================================================
# SECTION 1: SETUP AND DEPENDENCIES
# =============================================================================

"""
# Section 1: Setup and Dependencies

This section imports all necessary libraries for our retriever evaluation pipeline:
- **Ragas**: For synthetic data generation and evaluation metrics
- **LangChain**: For document loading, text splitting, and retriever implementations
- **Performance Tracking**: Libraries for measuring latency and resource usage
- **Visualization**: Matplotlib/Plotly for creating comparison charts
- **LangSmith**: For cost and latency tracking (optional but recommended)

We'll also set up our environment variables and basic configurations.
"""

# Core imports
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.storage import InMemoryStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Ragas imports
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    context_relevancy,
    answer_relevancy,
    faithfulness
)

# Qdrant and performance tracking
from qdrant_client import QdrantClient, models
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# LangSmith (optional)
try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("LangSmith not available. Performance tracking will be limited.")

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

print("✅ All dependencies imported successfully!")
print(f"📊 LangSmith available: {LANGSMITH_AVAILABLE}")

# =============================================================================
# SECTION 2: DOCUMENT LOADING AND PREPARATION
# =============================================================================

"""
# Section 2: Document Loading and Preparation

Here we load and prepare the "howpeopleuseai.pdf" document for evaluation:
- **PDF Loading**: Use PyPDFLoader to extract text from the PDF
- **Text Splitting**: Apply RecursiveCharacterTextSplitter for consistent chunking
- **Document Inspection**: Examine the loaded content to understand the data structure
- **Preprocessing**: Clean and prepare documents for vector store creation

This creates our base document corpus that all retrievers will work with.
"""

def load_and_prepare_documents(pdf_path: str = "./data/howpeopleuseai.pdf") -> Tuple[List, List]:
    """
    Load PDF document and prepare it for retrieval evaluation.
    
    Returns:
        Tuple of (original_documents, chunked_documents)
    """
    print(f"📄 Loading PDF from: {pdf_path}")
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"📊 Loaded {len(documents)} pages from PDF")
    print(f"📝 Total characters: {sum(len(doc.page_content) for doc in documents):,}")
    
    # Set up text splitter for standard chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split documents
    chunked_documents = text_splitter.split_documents(documents)
    
    print(f"🔪 Created {len(chunked_documents)} chunks")
    print(f"📏 Average chunk size: {np.mean([len(doc.page_content) for doc in chunked_documents]):.0f} characters")
    
    # Display sample chunk
    if chunked_documents:
        print("\n📋 Sample chunk:")
        print("-" * 50)
        print(chunked_documents[0].page_content[:300] + "...")
        print("-" * 50)
    
    return documents, chunked_documents

# Execute document loading
original_docs, standard_chunks = load_and_prepare_documents()

# =============================================================================
# SECTION 3: GOLDEN DATASET CREATION (10 QUESTIONS)
# =============================================================================

"""
# Section 3: Synthetic Test Dataset Generation

Using Ragas TestsetGenerator, we create a high-quality evaluation dataset:
- **Question Generation**: Generate 10 diverse questions from the PDF content
- **Question Types**: Include simple factual, reasoning, and multi-context questions
- **Ground Truth**: Create reference answers for each question
- **Quality Control**: Ensure questions are answerable from the document content
- **Dataset Export**: Save the dataset for reproducible evaluation

This golden dataset will be used to evaluate all retriever methods consistently.
"""

def create_synthetic_dataset(documents: List, test_size: int = 10) -> pd.DataFrame:
    """
    Generate synthetic evaluation dataset using Ragas.
    
    Args:
        documents: List of document chunks
        test_size: Number of questions to generate
    
    Returns:
        DataFrame with questions, contexts, and ground truth answers
    """
    print(f"🎯 Generating {test_size} synthetic questions...")
    
    # Initialize embeddings and LLM for generation
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create test set generator
    generator = TestsetGenerator.from_langchain(
        generator_llm=llm,
        critic_llm=llm,
        embeddings=embeddings
    )
    
    # Generate test set with different question types
    testset = generator.generate_with_langchain_docs(
        documents=documents[:20],  # Use first 20 chunks for generation
        test_size=test_size,
        distributions={
            simple: 0.4,      # 40% simple factual questions
            reasoning: 0.4,   # 40% reasoning questions
            multi_context: 0.2 # 20% multi-context questions
        }
    )
    
    # Convert to DataFrame
    test_df = testset.to_pandas()
    
    print(f"✅ Generated {len(test_df)} questions successfully!")
    print(f"📊 Question types distribution:")
    if 'evolution_type' in test_df.columns:
        print(test_df['evolution_type'].value_counts())
    
    # Display sample questions
    print("\n📝 Sample questions:")
    for i, row in test_df.head(3).iterrows():
        print(f"\nQ{i+1}: {row['question']}")
        print(f"A{i+1}: {row['ground_truth'][:100]}...")
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_dataset_{timestamp}.csv"
    test_df.to_csv(filename, index=False)
    print(f"💾 Dataset saved as: {filename}")
    
    return test_df

# Generate synthetic dataset
evaluation_dataset = create_synthetic_dataset(standard_chunks, test_size=10)

# =============================================================================
# SECTION 4: RETRIEVER COLLECTION SETUP
# =============================================================================

"""
# Section 4: Retriever Implementation and Organization

Set up all retriever methods from the notebook for systematic evaluation:
- **Naive Retriever**: Basic similarity search with embeddings
- **BM25 Retriever**: Keyword-based sparse retrieval
- **Multi-Query Retriever**: Multiple query reformulations
- **Parent Document Retriever**: Small-to-big retrieval strategy
- **Contextual Compression**: Reranking with Cohere
- **Ensemble Retriever**: Combination of multiple methods

Each retriever is configured with consistent parameters (k=10) for fair comparison.
"""

def setup_retrievers(documents: List, k: int = 10) -> Dict[str, Any]:
    """
    Set up all retriever methods for evaluation.
    
    Args:
        documents: List of document chunks
        k: Number of documents to retrieve
    
    Returns:
        Dictionary of retriever name -> retriever object
    """
    print(f"🔧 Setting up retrievers with k={k}...")
    
    retrievers = {}
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # 1. Naive Retriever (Embeddings-based)
    print("📍 Setting up Naive Retriever...")
    naive_vectorstore = Qdrant.from_documents(
        documents,
        embeddings,
        location=":memory:",
        collection_name="naive_retrieval"
    )
    retrievers["Naive"] = naive_vectorstore.as_retriever(search_kwargs={"k": k})
    
    # 2. BM25 Retriever
    print("📍 Setting up BM25 Retriever...")
    retrievers["BM25"] = BM25Retriever.from_documents(documents)
    retrievers["BM25"].k = k
    
    # 3. Multi-Query Retriever
    print("📍 Setting up Multi-Query Retriever...")
    retrievers["MultiQuery"] = MultiQueryRetriever.from_llm(
        retriever=retrievers["Naive"], 
        llm=llm
    )
    
    # 4. Parent Document Retriever
    print("📍 Setting up Parent Document Retriever...")
    # Create new vectorstore for parent document retriever
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="parent_documents",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )
    
    parent_vectorstore = QdrantVectorStore(
        collection_name="parent_documents",
        embedding=embeddings,
        client=client
    )
    
    store = InMemoryStore()
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    parent_retriever = ParentDocumentRetriever(
        vectorstore=parent_vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )
    parent_retriever.add_documents(documents)
    retrievers["ParentDocument"] = parent_retriever
    
    # 5. Contextual Compression (Reranking)
    print("📍 Setting up Contextual Compression Retriever...")
    compressor = CohereRerank(model="rerank-v3.5")
    retrievers["ContextualCompression"] = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retrievers["Naive"]
    )
    
    # 6. Ensemble Retriever
    print("📍 Setting up Ensemble Retriever...")
    ensemble_retrievers = [
        retrievers["BM25"],
        retrievers["Naive"],
        retrievers["MultiQuery"]
    ]
    weights = [1/len(ensemble_retrievers)] * len(ensemble_retrievers)
    
    retrievers["Ensemble"] = EnsembleRetriever(
        retrievers=ensemble_retrievers,
        weights=weights
    )
    
    print(f"✅ Successfully set up {len(retrievers)} retrievers!")
    for name in retrievers.keys():
        print(f"  - {name}")
    
    return retrievers

# Set up all retrievers
standard_retrievers = setup_retrievers(standard_chunks, k=10)

# =============================================================================
# SECTION 5: LANGSMITH INTEGRATION AND TRACKING
# =============================================================================

"""
# Section 5: Performance Monitoring Setup

Configure comprehensive performance tracking:
- **LangSmith Integration**: Set up project for cost and latency monitoring
- **Custom Metrics**: Implement timing decorators for execution measurement
- **Resource Tracking**: Monitor API calls and token usage
- **Tracing Setup**: Enable detailed operation tracing for each retriever

This ensures we capture accurate performance data for our analysis.
"""

class PerformanceTracker:
    """Track performance metrics for retriever evaluation."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.langsmith_client = None
        
        if LANGSMITH_AVAILABLE:
            try:
                self.langsmith_client = Client()
                print("✅ LangSmith client initialized")
            except Exception as e:
                print(f"⚠️ LangSmith initialization failed: {e}")
    
    def time_operation(self, operation_name: str):
        """Decorator to time operations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                duration = end_time - start_time
                self.metrics[f"{operation_name}_latency"].append(duration)
                
                return result
            return wrapper
        return decorator
    
    def record_metric(self, metric_name: str, value: float):
        """Record a custom metric."""
        self.metrics[metric_name].append(value)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        return summary

# Initialize performance tracker
performance_tracker = PerformanceTracker()

print("📊 Performance tracking initialized!")

# =============================================================================
# SECTION 6: RAGAS METRICS CONFIGURATION
# =============================================================================

"""
# Section 6: Evaluation Metrics Setup

Configure Ragas metrics specifically for retriever evaluation:
- **Context Precision**: Measures relevance of retrieved context
- **Context Recall**: Measures completeness of retrieved information
- **Context Relevancy**: Evaluates semantic relevance of retrieved chunks
- **Answer Relevancy**: Assesses quality of generated responses
- **Faithfulness**: Measures factual consistency with source material

These metrics provide comprehensive evaluation of retrieval quality.
"""

def setup_evaluation_metrics():
    """Configure Ragas metrics for evaluation."""
    
    metrics = [
        context_precision,
        context_recall,
        context_relevancy,
        answer_relevancy,
        faithfulness
    ]
    
    print("📏 Configured evaluation metrics:")
    for metric in metrics:
        print(f"  - {metric.name}")
    
    return metrics

# Set up evaluation metrics
evaluation_metrics = setup_evaluation_metrics()

# RAG Chain Template
RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

def create_rag_chain(retriever, llm=None):
    """Create a RAG chain with the given retriever."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini")
    
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    
    chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | llm, "context": itemgetter("context")}
    )
    
    return chain

print("🔗 RAG chain template configured!")

# =============================================================================
# SECTION 7: INDIVIDUAL RETRIEVER EVALUATION (5 STANDARD + PARENT DOCUMENT)
# =============================================================================

"""
# Section 7: Systematic Retriever Evaluation

Execute comprehensive evaluation for each retriever:
- **Automated Testing**: Run all 10 questions through each retriever
- **Metric Collection**: Gather Ragas scores for each retriever-question pair
- **Performance Measurement**: Record latency, cost, and resource usage
- **Error Handling**: Capture and log any evaluation failures
- **Result Storage**: Organize results in structured format for analysis

This creates our core evaluation dataset for comparison.
"""

def evaluate_retriever(retriever_name: str, retriever, test_dataset: pd.DataFrame, 
                      metrics: List, tracker: PerformanceTracker) -> Dict[str, Any]:
    """
    Evaluate a single retriever on the test dataset.
    
    Args:
        retriever_name: Name of the retriever
        retriever: Retriever object
        test_dataset: DataFrame with test questions
        metrics: List of Ragas metrics
        tracker: Performance tracker
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"🔍 Evaluating {retriever_name} retriever...")
    
    # Create RAG chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    rag_chain = create_rag_chain(retriever, llm)
    
    # Prepare evaluation data
    questions = test_dataset['question'].tolist()
    ground_truths = test_dataset['ground_truth'].tolist()
    
    # Generate answers and collect contexts
    answers = []
    contexts = []
    latencies = []
    
    for question in questions:
        start_time = time.time()
        try:
            result = rag_chain.invoke({"question": question})
            answer = result["response"].content
            context = [doc.page_content for doc in result["context"]]
            
            answers.append(answer)
            contexts.append(context)
            
        except Exception as e:
            print(f"⚠️ Error processing question: {e}")
            answers.append("Error occurred during processing")
            contexts.append([""])
        
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
        tracker.record_metric(f"{retriever_name}_latency", latency)
    
    # Create evaluation dataset
    eval_dataset = {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    }
    
    eval_df = pd.DataFrame(eval_dataset)
    
    # Run Ragas evaluation
    try:
        ragas_results = evaluate(eval_df, metrics=metrics)
        results = {
            'retriever_name': retriever_name,
            'ragas_scores': ragas_results,
            'avg_latency': np.mean(latencies),
            'total_latency': np.sum(latencies),
            'eval_dataset': eval_df
        }
        
        print(f"✅ {retriever_name} evaluation completed!")
        print(f"📊 Average latency: {np.mean(latencies):.2f}s")
        
    except Exception as e:
        print(f"❌ Ragas evaluation failed for {retriever_name}: {e}")
        results = {
            'retriever_name': retriever_name,
            'ragas_scores': None,
            'avg_latency': np.mean(latencies),
            'total_latency': np.sum(latencies),
            'eval_dataset': eval_df,
            'error': str(e)
        }
    
    return results

def evaluate_all_standard_retrievers(retrievers: Dict, test_dataset: pd.DataFrame, 
                                   metrics: List, tracker: PerformanceTracker) -> Dict[str, Any]:
    """Evaluate all standard retrievers."""
    print("🚀 Starting evaluation of all standard retrievers...")
    
    results = {}
    
    for retriever_name, retriever in retrievers.items():
        results[retriever_name] = evaluate_retriever(
            retriever_name, retriever, test_dataset, metrics, tracker
        )
        print(f"✅ Completed {retriever_name}")
        print("-" * 50)
    
    print("🎉 All standard retriever evaluations completed!")
    return results

# Run evaluation on standard retrievers
standard_results = evaluate_all_standard_retrievers(
    standard_retrievers, evaluation_dataset, evaluation_metrics, performance_tracker
)

# =============================================================================
# SECTION 8: SEMANTIC CHUNKING COMPARISON (5 RETRIEVERS WITH SEMANTIC CHUNKING)
# =============================================================================

"""
# Section 8: Semantic Chunking Analysis

Compare retriever performance with and without semantic chunking:
- **Semantic Chunking Implementation**: Apply semantic chunking to the PDF
- **Comparative Evaluation**: Test key retrievers on both chunking methods
- **Performance Delta**: Measure improvement/degradation from semantic chunking
- **Use Case Analysis**: Identify when semantic chunking helps vs. hurts

This analysis helps determine the value of semantic chunking for this document type.
"""

def create_semantic_chunks(documents: List) -> List:
    """Create semantic chunks from documents."""
    print("🧠 Creating semantic chunks...")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    
    # Apply semantic chunking to first 20 documents for efficiency
    semantic_documents = semantic_chunker.split_documents(documents[:20])
    
    print(f"🔪 Created {len(semantic_documents)} semantic chunks")
    print(f"📏 Average semantic chunk size: {np.mean([len(doc.page_content) for doc in semantic_documents]):.0f} characters")
    
    return semantic_documents

def setup_semantic_retrievers(semantic_docs: List, k: int = 10) -> Dict[str, Any]:
    """Set up retrievers with semantic chunking (excluding Parent Document)."""
    print(f"🧠 Setting up semantic retrievers with k={k}...")
    
    retrievers = {}
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # 1. Naive Retriever with Semantic Chunking
    print("📍 Setting up Semantic Naive Retriever...")
    semantic_vectorstore = Qdrant.from_documents(
        semantic_docs,
        embeddings,
        location=":memory:",
        collection_name="semantic_naive"
    )
    retrievers["Semantic_Naive"] = semantic_vectorstore.as_retriever(search_kwargs={"k": k})
    
    # 2. BM25 with Semantic Chunking
    print("📍 Setting up Semantic BM25 Retriever...")
    retrievers["Semantic_BM25"] = BM25Retriever.from_documents(semantic_docs)
    retrievers["Semantic_BM25"].k = k
    
    # 3. Multi-Query with Semantic Chunking
    print("📍 Setting up Semantic Multi-Query Retriever...")
    retrievers["Semantic_MultiQuery"] = MultiQueryRetriever.from_llm(
        retriever=retrievers["Semantic_Naive"], 
        llm=llm
    )
    
    # 4. Contextual Compression with Semantic Chunking
    print("📍 Setting up Semantic Contextual Compression Retriever...")
    compressor = CohereRerank(model="rerank-v3.5")
    retrievers["Semantic_ContextualCompression"] = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retrievers["Semantic_Naive"]
    )
    
    # 5. Ensemble with Semantic Chunking
    print("📍 Setting up Semantic Ensemble Retriever...")
    ensemble_retrievers = [
        retrievers["Semantic_BM25"],
        retrievers["Semantic_Naive"],
        retrievers["Semantic_MultiQuery"]
    ]
    weights = [1/len(ensemble_retrievers)] * len(ensemble_retrievers)
    
    retrievers["Semantic_Ensemble"] = EnsembleRetriever(
        retrievers=ensemble_retrievers,
        weights=weights
    )
    
    print(f"✅ Successfully set up {len(retrievers)} semantic retrievers!")
    for name in retrievers.keys():
        print(f"  - {name}")
    
    return retrievers

# Create semantic chunks and retrievers
semantic_chunks = create_semantic_chunks(original_docs)
semantic_retrievers = setup_semantic_retrievers(semantic_chunks, k=10)

# Evaluate semantic retrievers
semantic_results = evaluate_all_standard_retrievers(
    semantic_retrievers, evaluation_dataset, evaluation_metrics, performance_tracker
)

# =============================================================================
# SECTION 9: RESULTS COMPILATION AND ANALYSIS (11 TOTAL CONFIGURATIONS)
# =============================================================================

"""
# Section 9: Comprehensive Results Analysis

Aggregate and analyze all evaluation results:
- **Data Aggregation**: Combine metrics from all retrievers and tests
- **Statistical Analysis**: Calculate means, standard deviations, and rankings
- **Cost-Performance Analysis**: Evaluate efficiency trade-offs
- **Latency Analysis**: Compare response times across methods
- **Performance Rankings**: Rank retrievers by different criteria

This provides the foundation for our final recommendations.
"""

def compile_all_results(standard_results: Dict, semantic_results: Dict, 
                       tracker: PerformanceTracker) -> pd.DataFrame:
    """Compile all evaluation results into a comprehensive DataFrame."""
    print("📊 Compiling all evaluation results...")
    
    all_results = []
    
    # Process standard results
    for retriever_name, result in standard_results.items():
        if result['ragas_scores'] is not None:
            row = {
                'retriever_name': retriever_name,
                'chunking_type': 'Standard',
                'avg_latency': result['avg_latency'],
                'total_latency': result['total_latency']
            }
            
            # Add Ragas metrics
            for metric_name, score in result['ragas_scores'].items():
                row[metric_name] = score
            
            all_results.append(row)
    
    # Process semantic results
    for retriever_name, result in semantic_results.items():
        if result['ragas_scores'] is not None:
            row = {
                'retriever_name': retriever_name,
                'chunking_type': 'Semantic',
                'avg_latency': result['avg_latency'],
                'total_latency': result['total_latency']
            }
            
            # Add Ragas metrics
            for metric_name, score in result['ragas_scores'].items():
                row[metric_name] = score
            
            all_results.append(row)
    
    results_df = pd.DataFrame(all_results)
    
    # Calculate composite scores
    if len(results_df) > 0:
        metric_columns = [col for col in results_df.columns 
                         if col not in ['retriever_name', 'chunking_type', 'avg_latency', 'total_latency']]
        
        if metric_columns:
            results_df['composite_score'] = results_df[metric_columns].mean(axis=1)
            results_df['efficiency_score'] = results_df['composite_score'] / results_df['avg_latency']
    
    print(f"✅ Compiled results for {len(results_df)} retriever configurations")
    return results_df

def analyze_performance_differences(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance differences between standard and semantic chunking."""
    print("🔍 Analyzing performance differences...")
    
    analysis = {}
    
    # Get base retriever names (without Semantic_ prefix)
    semantic_retrievers = results_df[results_df['chunking_type'] == 'Semantic']['retriever_name'].tolist()
    base_names = [name.replace('Semantic_', '') for name in semantic_retrievers]
    
    comparisons = []
    
    for base_name in base_names:
        standard_row = results_df[
            (results_df['retriever_name'] == base_name) & 
            (results_df['chunking_type'] == 'Standard')
        ]
        semantic_row = results_df[
            (results_df['retriever_name'] == f'Semantic_{base_name}') & 
            (results_df['chunking_type'] == 'Semantic')
        ]
        
        if len(standard_row) > 0 and len(semantic_row) > 0:
            comparison = {
                'retriever': base_name,
                'standard_composite': standard_row['composite_score'].iloc[0],
                'semantic_composite': semantic_row['composite_score'].iloc[0],
                'standard_latency': standard_row['avg_latency'].iloc[0],
                'semantic_latency': semantic_row['avg_latency'].iloc[0]
            }
            
            comparison['composite_improvement'] = (
                comparison['semantic_composite'] - comparison['standard_composite']
            ) / comparison['standard_composite'] * 100
            
            comparison['latency_change'] = (
                comparison['semantic_latency'] - comparison['standard_latency']
            ) / comparison['standard_latency'] * 100
            
            comparisons.append(comparison)
    
    analysis['comparisons'] = pd.DataFrame(comparisons)
    analysis['best_overall'] = results_df.loc[results_df['composite_score'].idxmax()]
    analysis['fastest'] = results_df.loc[results_df['avg_latency'].idxmin()]
    analysis['most_efficient'] = results_df.loc[results_df['efficiency_score'].idxmax()]
    
    return analysis

# Compile and analyze results
compiled_results = compile_all_results(standard_results, semantic_results, performance_tracker)
performance_analysis = analyze_performance_differences(compiled_results)

# Display summary
print("\n📈 EVALUATION SUMMARY")
print("=" * 50)
print(f"Total configurations evaluated: {len(compiled_results)}")
print(f"Best overall performer: {performance_analysis['best_overall']['retriever_name']} ({performance_analysis['best_overall']['chunking_type']})")
print(f"Fastest retriever: {performance_analysis['fastest']['retriever_name']} ({performance_analysis['fastest']['chunking_type']})")
print(f"Most efficient: {performance_analysis['most_efficient']['retriever_name']} ({performance_analysis['most_efficient']['chunking_type']})")

# Save compiled results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
compiled_results.to_csv(f"compiled_results_{timestamp}.csv", index=False)
print(f"💾 Results saved as: compiled_results_{timestamp}.csv")

# =============================================================================
# SECTION 10: VISUALIZATION AND FINAL RECOMMENDATIONS
# =============================================================================

"""
# Section 10: Results Visualization and Conclusions

Create comprehensive visualizations and final analysis:
- **Performance Charts**: Bar charts comparing Ragas metrics across retrievers
- **Cost vs. Performance**: Scatter plots showing efficiency trade-offs
- **Latency Comparison**: Response time analysis across methods
- **Recommendation Matrix**: Best retriever for different use cases
- **Final Paragraph**: Detailed recommendation based on cost, latency, and performance

This section delivers actionable insights for retriever selection.
"""

def create_performance_visualizations(results_df: pd.DataFrame, analysis: Dict[str, Any]):
    """Create comprehensive visualizations of retriever performance."""
    print("📊 Creating performance visualizations...")
    
    # Set up the plotting environment
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Retriever Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Composite Score Comparison
    ax1 = axes[0, 0]
    if 'composite_score' in results_df.columns:
        results_sorted = results_df.sort_values('composite_score', ascending=True)
        bars = ax1.barh(range(len(results_sorted)), results_sorted['composite_score'])
        ax1.set_yticks(range(len(results_sorted)))
        ax1.set_yticklabels([f"{row['retriever_name']}\n({row['chunking_type']})" 
                            for _, row in results_sorted.iterrows()], fontsize=8)
        ax1.set_xlabel('Composite Score')
        ax1.set_title('Overall Performance Ranking')
        
        # Color bars by chunking type
        for i, (_, row) in enumerate(results_sorted.iterrows()):
            if row['chunking_type'] == 'Semantic':
                bars[i].set_color('lightcoral')
            else:
                bars[i].set_color('lightblue')
    
    # 2. Latency Comparison
    ax2 = axes[0, 1]
    results_latency = results_df.sort_values('avg_latency', ascending=True)
    bars2 = ax2.barh(range(len(results_latency)), results_latency['avg_latency'])
    ax2.set_yticks(range(len(results_latency)))
    ax2.set_yticklabels([f"{row['retriever_name']}\n({row['chunking_type']})" 
                        for _, row in results_latency.iterrows()], fontsize=8)
    ax2.set_xlabel('Average Latency (seconds)')
    ax2.set_title('Response Time Comparison')
    
    # Color bars by chunking type
    for i, (_, row) in enumerate(results_latency.iterrows()):
        if row['chunking_type'] == 'Semantic':
            bars2[i].set_color('lightcoral')
        else:
            bars2[i].set_color('lightblue')
    
    # 3. Efficiency Score (Performance vs Latency)
    ax3 = axes[1, 0]
    if 'efficiency_score' in results_df.columns:
        for chunking_type in results_df['chunking_type'].unique():
            subset = results_df[results_df['chunking_type'] == chunking_type]
            ax3.scatter(subset['avg_latency'], subset['composite_score'], 
                       label=chunking_type, alpha=0.7, s=100)
        
        ax3.set_xlabel('Average Latency (seconds)')
        ax3.set_ylabel('Composite Score')
        ax3.set_title('Performance vs Latency Trade-off')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Semantic vs Standard Comparison
    ax4 = axes[1, 1]
    if 'comparisons' in analysis and len(analysis['comparisons']) > 0:
        comp_df = analysis['comparisons']
        x_pos = range(len(comp_df))
        width = 0.35
        
        ax4.bar([x - width/2 for x in x_pos], comp_df['standard_composite'], 
               width, label='Standard', alpha=0.8)
        ax4.bar([x + width/2 for x in x_pos], comp_df['semantic_composite'], 
               width, label='Semantic', alpha=0.8)
        
        ax4.set_xlabel('Retriever Type')
        ax4.set_ylabel('Composite Score')
        ax4.set_title('Standard vs Semantic Chunking')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(comp_df['retriever'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"retriever_performance_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"📊 Visualizations saved as: retriever_performance_analysis_{timestamp}.png")
    
    plt.show()

def generate_detailed_recommendations(results_df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
    """Generate detailed recommendations based on evaluation results."""
    print("📝 Generating detailed recommendations...")
    
    recommendations = []
    
    # Overall best performer
    if 'best_overall' in analysis:
        best = analysis['best_overall']
        recommendations.append(f"**Best Overall Performer**: {best['retriever_name']} with {best['chunking_type']} chunking achieved the highest composite score of {best['composite_score']:.3f}.")
    
    # Fastest retriever
    if 'fastest' in analysis:
        fastest = analysis['fastest']
        recommendations.append(f"**Fastest Retriever**: {fastest['retriever_name']} with {fastest['chunking_type']} chunking had the lowest latency at {fastest['avg_latency']:.2f} seconds per query.")
    
    # Most efficient
    if 'most_efficient' in analysis:
        efficient = analysis['most_efficient']
        recommendations.append(f"**Most Efficient**: {efficient['retriever_name']} with {efficient['chunking_type']} chunking provided the best performance-to-latency ratio with an efficiency score of {efficient['efficiency_score']:.3f}.")
    
    # Semantic chunking analysis
    if 'comparisons' in analysis and len(analysis['comparisons']) > 0:
        comp_df = analysis['comparisons']
        avg_improvement = comp_df['composite_improvement'].mean()
        avg_latency_change = comp_df['latency_change'].mean()
        
        if avg_improvement > 5:
            recommendations.append(f"**Semantic Chunking Benefits**: On average, semantic chunking improved performance by {avg_improvement:.1f}% but increased latency by {avg_latency_change:.1f}%.")
        elif avg_improvement < -5:
            recommendations.append(f"**Semantic Chunking Drawbacks**: On average, semantic chunking decreased performance by {abs(avg_improvement):.1f}% and increased latency by {avg_latency_change:.1f}%.")
        else:
            recommendations.append(f"**Semantic Chunking Impact**: Semantic chunking showed mixed results with an average performance change of {avg_improvement:.1f}% and latency change of {avg_latency_change:.1f}%.")
    
    # Use case recommendations
    recommendations.append("\n**Use Case Recommendations**:")
    recommendations.append("- **For Speed-Critical Applications**: Choose the fastest retriever identified above")
    recommendations.append("- **For Maximum Accuracy**: Use the best overall performer regardless of latency")
    recommendations.append("- **For Balanced Performance**: Select the most efficient retriever for optimal cost-performance ratio")
    recommendations.append("- **For Cost-Sensitive Applications**: Consider simpler retrievers like BM25 or Naive retrieval")
    
    # Final recommendation paragraph
    final_recommendation = "\n".join(recommendations)
    
    # Save recommendations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"retriever_recommendations_{timestamp}.txt", "w") as f:
        f.write(final_recommendation)
    
    print(f"📄 Recommendations saved as: retriever_recommendations_{timestamp}.txt")
    
    return final_recommendation

def display_final_summary(results_df: pd.DataFrame, analysis: Dict[str, Any], tracker: PerformanceTracker):
    """Display comprehensive final summary."""
    print("\n" + "="*80)
    print("🎯 FINAL EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n📊 **Evaluation Overview**:")
    print(f"   • Total retriever configurations tested: {len(results_df)}")
    print(f"   • Questions per retriever: {len(evaluation_dataset)}")
    print(f"   • Total evaluations performed: {len(results_df) * len(evaluation_dataset)}")
    
    if 'best_overall' in analysis:
        best = analysis['best_overall']
        print(f"\n🏆 **Top Performers**:")
        print(f"   • Best Overall: {best['retriever_name']} ({best['chunking_type']}) - Score: {best['composite_score']:.3f}")
    
    if 'fastest' in analysis:
        fastest = analysis['fastest']
        print(f"   • Fastest: {fastest['retriever_name']} ({fastest['chunking_type']}) - {fastest['avg_latency']:.2f}s")
    
    if 'most_efficient' in analysis:
        efficient = analysis['most_efficient']
        print(f"   • Most Efficient: {efficient['retriever_name']} ({efficient['chunking_type']}) - Score: {efficient['efficiency_score']:.3f}")
    
    # Performance tracking summary
    perf_summary = tracker.get_summary()
    if perf_summary:
        print(f"\n⏱️ **Performance Metrics**:")
        total_time = sum([metrics['mean'] * metrics['count'] for metrics in perf_summary.values() if 'latency' in metrics])
        print(f"   • Total evaluation time: {total_time:.1f} seconds")
        print(f"   • Average query processing time: {total_time / (len(results_df) * len(evaluation_dataset)):.2f}s")
    
    print("\n" + "="*80)

# Execute visualization and recommendations
create_performance_visualizations(compiled_results, performance_analysis)
final_recommendations = generate_detailed_recommendations(compiled_results, performance_analysis)
display_final_summary(compiled_results, performance_analysis, performance_tracker)

print("\n📋 **DETAILED RECOMMENDATIONS**:")
print(final_recommendations)

print("\n🎉 **EVALUATION PIPELINE COMPLETED SUCCESSFULLY!**")
print("📁 All results, visualizations, and recommendations have been saved to files.")
print("📊 You can now copy the relevant sections into your Jupyter notebook.")
