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
