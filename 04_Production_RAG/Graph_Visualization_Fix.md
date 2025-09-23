# How to Visualize the Enhanced Graph

## Issue
When you try to display `enhanced_graph_builder`, it only shows:
```
<langgraph.graph.state.StateGraph at 0x12f1b90a0>
```

## Solution
You need to **compile the graph first** and then use the proper visualization methods. Here's the correct approach:

```python
# 1. First, compile the graph
enhanced_graph = enhanced_graph_builder.compile()

# 2. Then visualize it using one of these methods:

# Method 1: ASCII representation (works in any environment)
enhanced_graph.get_graph().print_ascii()

# Method 2: If you want to see the graph structure
print("Graph nodes:", enhanced_graph.get_graph().nodes)
print("Graph edges:", enhanced_graph.get_graph().edges)

# Method 3: For Jupyter notebooks with graphviz installed
# enhanced_graph.get_graph().draw_mermaid()  # Requires mermaid
```

## Complete Working Example

```python
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from langchain_core.documents import Document
from typing import Literal
import numpy as np

# Enhanced state definition
class EnhancedState(TypedDict):
    question: str
    context: list[Document]
    response: str
    context_quality: float
    has_sufficient_context: bool

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
    similarities = []
    for ctx_emb in context_embeddings:
        similarity = np.dot(question_embedding, ctx_emb) / (np.linalg.norm(question_embedding) * np.linalg.norm(ctx_emb))
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

# Build the enhanced graph
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

# IMPORTANT: Compile the graph first!
enhanced_graph = enhanced_graph_builder.compile()

# Now you can visualize it
print("Enhanced Graph Structure:")
enhanced_graph.get_graph().print_ascii()

# You can also inspect the components
print("\nGraph Nodes:", list(enhanced_graph.get_graph().nodes.keys()))
print("Graph Edges:", enhanced_graph.get_graph().edges)
```

## Expected ASCII Output
When you run `enhanced_graph.get_graph().print_ascii()`, you should see something like:

```
           +-----------+
           |   START   |
           +-----------+
                 |
                 v
           +-----------+
           |  retrieve |
           +-----------+
                 |
                 v
      +------------------+
      | validate_context |
      +------------------+
            /        \
           /          \
          v            v
   +-----------+  +------------------+
   | generate  |  | fallback_response|
   +-----------+  +------------------+
          |              |
          v              v
       +-----+        +-----+
       | END |        | END |
       +-----+        +-----+
```

## Key Points
1. **Always compile first**: `enhanced_graph = enhanced_graph_builder.compile()`
2. **Use proper visualization**: `enhanced_graph.get_graph().print_ascii()`
3. **The builder object itself cannot be visualized** - only the compiled graph can be
