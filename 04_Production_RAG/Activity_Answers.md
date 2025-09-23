# Activity #1 Answers: Advanced Document Chunking Strategies

## Assignment 1 - Activity #1

**Task**: Brainstorm ideas for splitting large single documents into smaller documents that would be more sensitive to specific data formats than the naive chunking approach used in the notebook.

---

## Question #1: Embedding Dimension

**Question**: What is the embedding dimension, given that we're using `embeddinggemma`?

**Answer**: The embedding dimension for `embeddinggemma:latest` is **768**.

This can be verified by running:
```bash
ollama show embeddinggemma:latest
```

The output shows:
```
Model
    architecture        gemma3
    parameters          307.58M
    context length      2048
    embedding length    768
    quantization        BF16
```

Therefore, `embedding_dim = 768`

---

## 1. Semantic-Based Chunking

**Concept**: Use semantic understanding to identify natural breakpoints instead of splitting based purely on character count. This approach maintains semantic coherence within each chunk.

**Code Example**:

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document

class SemanticTextSplitter(TextSplitter):
    def __init__(self, model_name="all-MiniLM-L6-v2", similarity_threshold=0.7, max_chunk_size=1000):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
    
    def split_text(self, text: str) -> list[str]:
        # Split into sentences
        sentences = text.split('. ')
        
        # Get embeddings for each sentence
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for i, sentence in enumerate(sentences):
            # Check semantic similarity with previous sentence
            if i > 0:
                similarity = cosine_similarity(
                    [embeddings[i-1]], [embeddings[i]]
                )[0][0]
                
                # If similarity drops below threshold or chunk is too large, start new chunk
                if (similarity < self.similarity_threshold or 
                    current_chunk_size + len(sentence) > self.max_chunk_size):
                    chunks.append('. '.join(current_chunk))
                    current_chunk = [sentence]
                    current_chunk_size = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_chunk_size += len(sentence)
            else:
                current_chunk.append(sentence)
                current_chunk_size = len(sentence)
        
        # Add the last chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks

# Usage example
semantic_splitter = SemanticTextSplitter(similarity_threshold=0.6)
chunks = semantic_splitter.split_documents([document])
```

---

## 2. Document Structure-Aware Chunking

**Concept**: Leverage the inherent structure of documents (headings, sections, paragraphs) to create more meaningful chunks that preserve hierarchical relationships and context.

**Code Example**:

```python
import re
from typing import List, Dict
from langchain_core.documents import Document

class StructureAwareChunker:
    def __init__(self, max_chunk_size=1000):
        self.max_chunk_size = max_chunk_size
        
        # Patterns for different document structures
        self.heading_patterns = {
            'h1': r'^#\s+(.+)$',
            'h2': r'^##\s+(.+)$', 
            'h3': r'^###\s+(.+)$',
            'section': r'^(\d+\.?\s+.+)$',
            'subsection': r'^(\d+\.\d+\.?\s+.+)$'
        }
    
    def extract_structure(self, text: str) -> List[Dict]:
        """Extract document structure with hierarchy"""
        lines = text.split('\n')
        structure = []
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a heading
            heading_found = False
            for level, pattern in self.heading_patterns.items():
                if re.match(pattern, line):
                    # Save previous section
                    if current_section and current_content:
                        current_section['content'] = '\n'.join(current_content)
                        structure.append(current_section)
                    
                    # Start new section
                    current_section = {
                        'level': level,
                        'title': line,
                        'content': ''
                    }
                    current_content = []
                    heading_found = True
                    break
            
            if not heading_found:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            current_section['content'] = '\n'.join(current_content)
            structure.append(current_section)
        
        return structure
    
    def chunk_by_structure(self, document: Document) -> List[Document]:
        """Create chunks based on document structure"""
        structure = self.extract_structure(document.page_content)
        chunks = []
        
        for section in structure:
            content = f"{section['title']}\n\n{section['content']}"
            
            # If section is too large, split it further
            if len(content) > self.max_chunk_size:
                # Split by paragraphs within the section
                paragraphs = section['content'].split('\n\n')
                current_chunk = section['title'] + '\n\n'
                
                for para in paragraphs:
                    if len(current_chunk + para) > self.max_chunk_size:
                        chunks.append(Document(
                            page_content=current_chunk.strip(),
                            metadata={
                                **document.metadata,
                                'section': section['title'],
                                'level': section['level']
                            }
                        ))
                        current_chunk = section['title'] + '\n\n' + para + '\n\n'
                    else:
                        current_chunk += para + '\n\n'
                
                if current_chunk.strip():
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata={
                            **document.metadata,
                            'section': section['title'],
                            'level': section['level']
                        }
                    ))
            else:
                chunks.append(Document(
                    page_content=content,
                    metadata={
                        **document.metadata,
                        'section': section['title'],
                        'level': section['level']
                    }
                ))
        
        return chunks

# Usage example
structure_chunker = StructureAwareChunker(max_chunk_size=800)
structured_chunks = structure_chunker.chunk_by_structure(document)
```

---

## 3. Content-Type Adaptive Chunking

**Concept**: Use different chunking strategies based on the type of content detected (code, tables, lists, figures). This preserves the integrity of different content types.

**Code Example**:

```python
import re
from typing import List, Tuple
from langchain_core.documents import Document

class ContentTypeAdaptiveChunker:
    def __init__(self, max_chunk_size=1000):
        self.max_chunk_size = max_chunk_size
        
        # Patterns for different content types
        self.patterns = {
            'code_block': r'```[\s\S]*?```',
            'table': r'\|.*\|[\s\S]*?\n\s*\n',
            'list': r'(?:^\s*[-*+]\s+.+\n?)+',
            'numbered_list': r'(?:^\s*\d+\.\s+.+\n?)+',
            'figure_caption': r'Figure\s+\d+[:\.].*?(?=\n\n|\n[A-Z]|\Z)',
            'citation': r'\[[\d,\s-]+\]|\(\w+\s+et\s+al\.,?\s+\d{4}\)'
        }
    
    def identify_content_type(self, text: str) -> List[Tuple[str, int, int, str]]:
        """Identify different content types and their positions"""
        content_blocks = []
        
        for content_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.MULTILINE):
                content_blocks.append((
                    content_type,
                    match.start(),
                    match.end(),
                    match.group()
                ))
        
        # Sort by position
        content_blocks.sort(key=lambda x: x[1])
        return content_blocks
    
    def chunk_by_content_type(self, document: Document) -> List[Document]:
        """Create chunks based on content type"""
        text = document.page_content
        content_blocks = self.identify_content_type(text)
        chunks = []
        
        if not content_blocks:
            # No special content found, use regular chunking
            return self._regular_chunk(document)
        
        last_end = 0
        current_chunk = ""
        
        for content_type, start, end, content in content_blocks:
            # Add text before this content block
            before_content = text[last_end:start].strip()
            
            # Check if we should start a new chunk
            if self._should_start_new_chunk(current_chunk, before_content, content, content_type):
                if current_chunk.strip():
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata=document.metadata
                    ))
                current_chunk = ""
            
            # Add the before content and the special content
            if before_content:
                current_chunk += before_content + "\n\n"
            
            current_chunk += self._format_content_by_type(content, content_type) + "\n\n"
            last_end = end
        
        # Add remaining text
        remaining_text = text[last_end:].strip()
        if remaining_text:
            if len(current_chunk + remaining_text) > self.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata=document.metadata
                    ))
                current_chunk = remaining_text
            else:
                current_chunk += remaining_text
        
        # Add the final chunk
        if current_chunk.strip():
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata=document.metadata
            ))
        
        return chunks
    
    def _should_start_new_chunk(self, current_chunk: str, before_content: str, 
                               special_content: str, content_type: str) -> bool:
        """Determine if we should start a new chunk"""
        total_length = len(current_chunk) + len(before_content) + len(special_content)
        
        # Always keep certain content types together
        if content_type in ['code_block', 'table']:
            return total_length > self.max_chunk_size and len(current_chunk) > 0
        
        return total_length > self.max_chunk_size
    
    def _format_content_by_type(self, content: str, content_type: str) -> str:
        """Format content based on its type"""
        if content_type == 'code_block':
            return f"[CODE BLOCK]\n{content}\n[/CODE BLOCK]"
        elif content_type == 'table':
            return f"[TABLE]\n{content}\n[/TABLE]"
        elif content_type in ['list', 'numbered_list']:
            return f"[LIST]\n{content}\n[/LIST]"
        elif content_type == 'figure_caption':
            return f"[FIGURE]\n{content}\n[/FIGURE]"
        else:
            return content
    
    def _regular_chunk(self, document: Document) -> List[Document]:
        """Fallback to regular chunking"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=100
        )
        return splitter.split_documents([document])

# Usage example
adaptive_chunker = ContentTypeAdaptiveChunker(max_chunk_size=800)
adaptive_chunks = adaptive_chunker.chunk_by_content_type(document)

# Example of processing different content types
for chunk in adaptive_chunks:
    print(f"Chunk length: {len(chunk.page_content)}")
    print(f"Content preview: {chunk.page_content[:100]}...")
    print("---")
```

---

## Benefits of These Approaches

1. **Semantic-Based Chunking**:
   - Maintains topical coherence
   - Reduces context fragmentation
   - Improves retrieval relevance

2. **Structure-Aware Chunking**:
   - Preserves document hierarchy
   - Maintains section context
   - Better for academic and technical documents

3. **Content-Type Adaptive Chunking**:
   - Preserves code integrity
   - Keeps tables and figures complete
   - Maintains list structure and relationships

These advanced chunking strategies would significantly improve the RAG system's ability to retrieve relevant and coherent context compared to the naive character-based approach used in the original notebook.


# question # 2
# Question #2: Extending LangGraph Implementation for Edge Cases

## Current Implementation Analysis

The current LangGraph implementation follows a simple linear flow:
1. **START** → **retrieve** → **generate** → **END**
2. State contains: `question`, `context`, `response`
3. No error handling or edge case management

## Proposed Extensions for Edge Cases

### 2.1 Handling No Relevant Context Found

**Problem**: When the retriever finds no relevant context, the system still proceeds to generation, potentially leading to hallucinated responses.

**Solution**: Add a context validation node and conditional routing.

#### Modified Graph Structure:
```
START → retrieve → validate_context → [conditional routing]
                        ↓
                   [has_context] → generate → END
                        ↓
                   [no_context] → fallback_response → END
```

#### Implementation:

```python
from typing import Literal
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from langchain_core.documents import Document

class EnhancedState(TypedDict):
    question: str
    context: list[Document]
    response: str
    context_quality: float  # New field for context relevance score
    has_sufficient_context: bool  # New field for routing decision

def validate_context(state: EnhancedState) -> EnhancedState:
    """Validate if retrieved context is sufficient and relevant."""
    context = state["context"]
    question = state["question"]
    
    if not context or len(context) == 0:
        return {
            "context_quality": 0.0,
            "has_sufficient_context": False
        }
    
    # Calculate relevance score using embedding similarity
    question_embedding = embedding_model.embed_query(question)
    context_embeddings = [embedding_model.embed_query(doc.page_content) for doc in context]
    
    # Calculate average cosine similarity
    from numpy import dot
    from numpy.linalg import norm
    
    similarities = []
    for ctx_emb in context_embeddings:
        similarity = dot(question_embedding, ctx_emb) / (norm(question_embedding) * norm(ctx_emb))
        similarities.append(similarity)
    
    avg_similarity = sum(similarities) / len(similarities)
    
    # Set threshold for sufficient context (adjustable)
    threshold = 0.3
    
    return {
        "context_quality": avg_similarity,
        "has_sufficient_context": avg_similarity >= threshold
    }

def fallback_response(state: EnhancedState) -> EnhancedState:
    """Generate response when no sufficient context is found."""
    fallback_message = (
        "I don't have sufficient relevant information in my knowledge base "
        "to answer your question about AI usage. Could you please rephrase "
        "your question or ask about a different aspect of how people use AI?"
    )
    return {"response": fallback_message}

def conditional_router(state: EnhancedState) -> Literal["generate", "fallback_response"]:
    """Route based on context sufficiency."""
    return "generate" if state["has_sufficient_context"] else "fallback_response"

# Modified graph construction
enhanced_graph_builder = StateGraph(EnhancedState)
enhanced_graph_builder.add_node("retrieve", retrieve)
enhanced_graph_builder.add_node("validate_context", validate_context)
enhanced_graph_builder.add_node("generate", generate)
enhanced_graph_builder.add_node("fallback_response", fallback_response)

# Add edges
enhanced_graph_builder.add_edge(START, "retrieve")
enhanced_graph_builder.add_edge("retrieve", "validate_context")
enhanced_graph_builder.add_conditional_edges(
    "validate_context",
    conditional_router,
    {
        "generate": "generate",
        "fallback_response": "fallback_response"
    }
)
enhanced_graph_builder.add_edge("generate", END)
enhanced_graph_builder.add_edge("fallback_response", END)
```

### 2.2 Handling Response Fact-Checking

**Problem**: Generated responses may contain inaccuracies or hallucinations even with good context.

**Solution**: Add a fact-checking node with verification and correction capabilities.

#### Extended Graph Structure:
```
START → retrieve → validate_context → generate → fact_check → [conditional routing]
                                                      ↓
                                              [verified] → END
                                                      ↓
                                              [needs_correction] → correct_response → END
```

#### Implementation:

```python
class FactCheckedState(TypedDict):
    question: str
    context: list[Document]
    response: str
    context_quality: float
    has_sufficient_context: bool
    fact_check_score: float  # New field
    needs_correction: bool   # New field
    corrected_response: str  # New field

def fact_check(state: FactCheckedState) -> FactCheckedState:
    """Fact-check the generated response against the context."""
    response = state["response"]
    context = state["context"]
    
    # Create a fact-checking prompt
    fact_check_prompt = ChatPromptTemplate.from_messages([
        ("human", """
        CONTEXT:
        {context}
        
        GENERATED_RESPONSE:
        {response}
        
        Please analyze if the generated response is factually accurate based on the provided context.
        Rate the accuracy on a scale of 0.0 to 1.0 where:
        - 1.0 = Completely accurate and supported by context
        - 0.7-0.9 = Mostly accurate with minor issues
        - 0.4-0.6 = Partially accurate but has some errors
        - 0.0-0.3 = Largely inaccurate or unsupported
        
        Respond with just the numerical score (e.g., 0.8).
        """)
    ])
    
    fact_check_chain = fact_check_prompt | ollama_chat_model | StrOutputParser()
    
    context_text = "\n".join([doc.page_content for doc in context])
    score_response = fact_check_chain.invoke({
        "context": context_text,
        "response": response
    })
    
    try:
        fact_check_score = float(score_response.strip())
    except ValueError:
        fact_check_score = 0.5  # Default to moderate confidence
    
    # Set threshold for acceptable accuracy
    accuracy_threshold = 0.7
    
    return {
        "fact_check_score": fact_check_score,
        "needs_correction": fact_check_score < accuracy_threshold
    }

def correct_response(state: FactCheckedState) -> FactCheckedState:
    """Generate a corrected response with explicit fact-checking."""
    context = state["context"]
    question = state["question"]
    original_response = state["response"]
    
    correction_prompt = ChatPromptTemplate.from_messages([
        ("human", """
        CONTEXT:
        {context}
        
        ORIGINAL_QUESTION:
        {question}
        
        ORIGINAL_RESPONSE (which may contain inaccuracies):
        {original_response}
        
        Please provide a corrected and more accurate response to the original question.
        Base your answer STRICTLY on the provided context. If any information cannot be
        verified from the context, explicitly state "This information is not available
        in the provided context."
        
        Be conservative and only make claims that are directly supported by the context.
        """)
    ])
    
    correction_chain = correction_prompt | ollama_chat_model | StrOutputParser()
    
    context_text = "\n".join([doc.page_content for doc in context])
    corrected_response = correction_chain.invoke({
        "context": context_text,
        "question": question,
        "original_response": original_response
    })
    
    return {"corrected_response": corrected_response}

def fact_check_router(state: FactCheckedState) -> Literal["finalize", "correct_response"]:
    """Route based on fact-check results."""
    return "correct_response" if state["needs_correction"] else "finalize"

def finalize(state: FactCheckedState) -> FactCheckedState:
    """Finalize the response (use corrected if available, otherwise original)."""
    if state.get("corrected_response"):
        final_response = state["corrected_response"]
    else:
        final_response = state["response"]
    
    return {"response": final_response}
```

## Complete Enhanced Graph Implementation

```python
# Complete enhanced graph with both edge case handlers
complete_enhanced_graph = StateGraph(FactCheckedState)

# Add all nodes
complete_enhanced_graph.add_node("retrieve", retrieve)
complete_enhanced_graph.add_node("validate_context", validate_context)
complete_enhanced_graph.add_node("generate", generate)
complete_enhanced_graph.add_node("fallback_response", fallback_response)
complete_enhanced_graph.add_node("fact_check", fact_check)
complete_enhanced_graph.add_node("correct_response", correct_response)
complete_enhanced_graph.add_node("finalize", finalize)

# Add edges and conditional routing
complete_enhanced_graph.add_edge(START, "retrieve")
complete_enhanced_graph.add_edge("retrieve", "validate_context")

# Route based on context quality
complete_enhanced_graph.add_conditional_edges(
    "validate_context",
    conditional_router,
    {
        "generate": "generate",
        "fallback_response": "fallback_response"
    }
)

# Fact-check generated responses
complete_enhanced_graph.add_edge("generate", "fact_check")

# Route based on fact-check results
complete_enhanced_graph.add_conditional_edges(
    "fact_check",
    fact_check_router,
    {
        "finalize": "finalize",
        "correct_response": "correct_response"
    }
)

# Finalize corrected responses
complete_enhanced_graph.add_edge("correct_response", "finalize")

# End paths
complete_enhanced_graph.add_edge("fallback_response", END)
complete_enhanced_graph.add_edge("finalize", END)

# Compile the enhanced graph
enhanced_rag_graph = complete_enhanced_graph.compile()
```

## Additional Enhancements

### 3. Query Clarification Node
For ambiguous queries, add a clarification node:

```python
def clarify_query(state: FactCheckedState) -> FactCheckedState:
    """Detect and handle ambiguous queries."""
    # Implementation for query ambiguity detection
    pass
```

### 4. Multi-Step Reasoning
For complex questions requiring multiple retrieval steps:

```python
def decompose_query(state: FactCheckedState) -> FactCheckedState:
    """Break complex queries into sub-questions."""
    # Implementation for query decomposition
    pass
```

### 5. Confidence Scoring
Add confidence scores to responses:

```python
def add_confidence_score(state: FactCheckedState) -> FactCheckedState:
    """Add confidence scoring to responses."""
    # Implementation for confidence estimation
    pass
```

## Benefits of This Approach

1. **Robustness**: Handles edge cases gracefully
2. **Transparency**: Users know when information is insufficient
3. **Accuracy**: Fact-checking reduces hallucinations
4. **Flexibility**: Easy to add more validation steps
5. **Maintainability**: Clear separation of concerns in nodes
6. **Observability**: Each step can be monitored and debugged

This enhanced implementation transforms the simple linear RAG pipeline into a robust, production-ready system that can handle real-world edge cases while maintaining the clarity and modularity that LangGraph provides.
