## ❓ Question #2: Alternative Approaches for Tool Control

### __1. Explicit Sequential Tool Chains__

```python
# Force a specific tool sequence using LCEL chains
tool_chain = tool1 | tool2 | tool3

# Or using RunnableSequence
from langchain_core.runnables import RunnableSequence
forced_sequence = RunnableSequence([search_tool, analysis_tool, write_tool])
```

### __2. State-Based Tool Gating__

```python
# Use state to control which tools are available
class ToolControlState(TypedDict):
    current_step: str
    available_tools: List[str]
    completed_tools: List[str]

def tool_gate_node(state):
    if state["current_step"] == "search":
        return {"available_tools": ["tavily_search"]}
    elif state["current_step"] == "analyze":
        return {"available_tools": ["analysis_tool"]}
```

### __3. Conditional Tool Routing__

```python
# Create explicit routing logic
def route_to_tools(state):
    if "search" in state["query"].lower():
        return "search_node"
    elif "analyze" in state["query"].lower():
        return "analysis_node"
    else:
        return "default_node"

# Add conditional edges with explicit routing
graph.add_conditional_edges(
    "router",
    route_to_tools,
    {"search_node": "search_node", "analysis_node": "analysis_node"}
)
```

### __4. Tool Dependency Enforcement__

```python
# Require certain tools to run before others
class DependencyState(TypedDict):
    search_completed: bool
    analysis_ready: bool

def analysis_node(state):
    if not state.get("search_completed", False):
        raise ValueError("Must complete search before analysis")
    # Proceed with analysis tool
```

### __5. Tool Validation Decorators__

```python
def require_tools(*required_tools):
    def decorator(func):
        def wrapper(state):
            # Check if required tools were used
            for tool in required_tools:
                if tool not in state.get("used_tools", []):
                    return {"error": f"Must use {tool} first"}
            return func(state)
        return wrapper
    return decorator

@require_tools("search_tool", "validation_tool")
def final_processing_node(state):
    # This node can only run after search_tool and validation_tool
    pass
```

### __6. Explicit Tool Orchestration__

```python
# Create a dedicated tool orchestrator
class ToolOrchestrator:
    def __init__(self, tool_sequence):
        self.sequence = tool_sequence
        self.current_step = 0
    
    def next_tool(self):
        if self.current_step < len(self.sequence):
            tool = self.sequence[self.current_step]
            self.current_step += 1
            return tool
        return None

# Use in a node
def orchestrated_node(state):
    orchestrator = state["tool_orchestrator"]
    next_tool = orchestrator.next_tool()
    if next_tool:
        return next_tool.invoke(state["query"])
```

### __7. Tool Access Control Lists (ACLs)__

```python
# Define which agents can use which tools
TOOL_PERMISSIONS = {
    "search_agent": ["tavily_search", "web_scraper"],
    "analysis_agent": ["data_analyzer", "summarizer"],
    "writer_agent": ["document_writer", "editor"]
}

def create_restricted_agent(agent_name, llm, system_prompt):
    allowed_tools = [tool for tool in ALL_TOOLS 
                    if tool.name in TOOL_PERMISSIONS[agent_name]]
    return create_agent(llm, allowed_tools, system_prompt)
```

### __8. Pipeline Enforcement__

```python
# Create a strict pipeline where each step must complete
class PipelineState(TypedDict):
    step: int
    pipeline_results: List[Any]

def pipeline_node(state):
    current_step = state["step"]
    if current_step == 0:
        # Must use search tool first
        result = search_tool.invoke(state["query"])
        return {"step": 1, "pipeline_results": [result]}
    elif current_step == 1:
        # Must use analysis tool second
        result = analysis_tool.invoke(state["pipeline_results"][0])
        return {"step": 2, "pipeline_results": state["pipeline_results"] + [result]}
```

### __9. Tool Workflow Templates__

```python
# Define reusable workflow templates
RESEARCH_WORKFLOW = [
    ("search", "tavily_search"),
    ("retrieve", "rag_search"), 
    ("analyze", "content_analyzer"),
    ("write", "document_writer")
]

def execute_workflow(workflow, state):
    results = []
    for step_name, tool_name in workflow:
        tool = TOOL_REGISTRY[tool_name]
        result = tool.invoke(state)
        results.append((step_name, result))
        state = update_state(state, step_name, result)
    return results
```

### __10. Guard Rails and Validation__

```python
# Add validation nodes between tool usage
def validate_search_results(state):
    if len(state["search_results"]) < 3:
        return {"error": "Insufficient search results, retry search"}
    return {"validated": True}

# Chain: search_tool → validate_search_results → analysis_tool
```

## __Summary of Approaches:__

1. __Restrictive__: Limit tool access per agent (what the notebook does)
2. __Sequential__: Force specific tool order using chains/pipelines
3. __Conditional__: Route based on state/content analysis
4. __Validation__: Check prerequisites before allowing tool usage
5. __Orchestration__: Use dedicated controllers for tool flow
6. __Templates__: Define reusable workflow patterns

The notebook uses approach #1 (restrictive tool assignment) combined with intelligent LLM-based routing, but these other methods provide even more explicit control when needed.
