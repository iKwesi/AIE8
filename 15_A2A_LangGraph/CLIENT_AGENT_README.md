# 🤖 Client Agent - A2A Protocol Demonstration

This directory contains a **LangGraph client agent** that demonstrates agent-to-agent communication via the A2A protocol.

## 📋 Overview

The client agent is a simple LangGraph agent with **one tool**: the ability to call your A2A server agent. This demonstrates the core concept of A2A - agents using other agents as tools.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER QUERY                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              CLIENT AGENT (LangGraph)                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Tool: call_a2a_agent                                │   │
│  │  - Fetches AgentCard from server                     │   │
│  │  - Sends query via A2A protocol                      │   │
│  │  - Returns server response                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP Request (A2A Protocol)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              A2A SERVER AGENT                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Tools:                                              │   │
│  │  - Tavily Web Search                                 │   │
│  │  - ArXiv Academic Papers                             │   │
│  │  - RAG Document Retrieval                            │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Helpfulness Evaluation Loop                         │   │
│  │  - Evaluates response quality                        │   │
│  │  - Iterates up to 10 times if needed                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
                  RESPONSE
```

## 📁 Files

- **`app/a2a_tool.py`**: LangChain tool that calls the A2A server
- **`app/client_agent.py`**: LangGraph client agent implementation
- **`app/test_client_agent.py`**: Comprehensive test suite

## 🚀 Quick Start

### 1. Start the A2A Server

In one terminal:

```bash
uv run python -m app
```

The server will start at `http://localhost:10000`

### 2. Run the Client Agent Tests

In another terminal:

```bash
# Run full test suite
uv run python app/test_client_agent.py

# OR: Run interactive demo
uv run python run_client_demo.py

# OR: Run quick demo (3 predefined queries)
uv run python run_client_demo.py --quick
```

This will run a comprehensive test suite demonstrating:
- Web search queries (Tavily)
- Academic paper search (ArXiv)
- Document retrieval (RAG)
- Multi-turn conversations
- Combined queries using multiple tools

## 🧪 Test Suite

The test suite includes 5 tests:

### Test 1: Web Search (Tavily)
```python
Query: "What are the latest developments in artificial intelligence in 2025?"
```
Demonstrates the client agent delegating a web search query to the server.

### Test 2: Academic Papers (ArXiv)
```python
Query: "Find recent papers on transformer architectures in machine learning"
```
Shows the server using ArXiv to find academic papers.

### Test 3: Document Retrieval (RAG)
```python
Query: "What information do the documents contain about how people use AI?"
```
Tests the RAG system retrieving information from local documents.

### Test 4: Multi-Turn Conversation
```python
Turn 1: "Find me papers about large language models"
Turn 2: "Can you summarize the key findings from those papers?"
Turn 3: "Which paper seems most relevant for understanding attention mechanisms?"
```
Demonstrates context preservation across multiple turns.

### Test 5: Combined Query
```python
Query: "Compare recent research papers on AI with current industry developments"
```
May trigger multiple tools on the server side.

## 💻 Using the Client Agent Programmatically

### Basic Usage

```python
import asyncio
from app.client_agent import run_client_agent_async

async def main():
    response = await run_client_agent_async(
        "What are the latest AI developments?",
        thread_id="my_conversation"
    )
    print(response)

asyncio.run(main())
```

### Multi-Turn Conversation

```python
import asyncio
from app.client_agent import run_client_agent_async

async def main():
    thread_id = "conversation_1"
    
    # First query
    response1 = await run_client_agent_async(
        "Find papers on transformers",
        thread_id=thread_id
    )
    
    # Follow-up (has context from first query)
    response2 = await run_client_agent_async(
        "Summarize the key findings",
        thread_id=thread_id
    )

asyncio.run(main())
```

### Building Your Own Client

```python
from app.client_agent import build_client_agent

# Build the agent
agent = build_client_agent()

# Use it with custom configuration
config = {"configurable": {"thread_id": "custom_thread"}}
inputs = {"messages": [("user", "Your query here")]}

for event in agent.stream(inputs, config, stream_mode="values"):
    # Process events
    pass
```

## 🔍 How It Works

### 1. A2A Tool (`a2a_tool.py`)

The `call_a2a_agent` tool:
1. Connects to the A2A server at `http://localhost:10000`
2. Fetches the server's AgentCard (metadata about capabilities)
3. Sends the user's query via A2A protocol
4. Extracts and returns the response

### 2. Client Agent (`client_agent.py`)

The client agent is a simple LangGraph with:
- **Agent Node**: LLM that decides when to use the A2A tool
- **Action Node**: Executes the A2A tool
- **Routing**: Continues until no more tool calls needed

### 3. A2A Protocol Flow

```
1. Client Agent receives user query
2. LLM decides to use call_a2a_agent tool
3. Tool fetches AgentCard from server
4. Tool sends query via A2A protocol
5. Server processes query (may use Tavily/ArXiv/RAG)
6. Server evaluates response helpfulness
7. Server returns response
8. Client Agent presents result to user
```

## 🎯 Key Concepts Demonstrated

### 1. Agent Composition
Instead of building one agent with all tools, we have:
- **Client Agent**: Simple orchestrator
- **Server Agent**: Specialized with multiple tools

### 2. A2A Protocol Benefits
- **Standardized Communication**: Both agents speak the same protocol
- **Interoperability**: Could swap server with any A2A-compliant agent
- **Separation of Concerns**: Client handles orchestration, server handles execution

### 3. Tool Abstraction
From the client's perspective, the entire server agent is just "a tool". This enables:
- Easy scaling (add more server instances)
- Easy replacement (swap with different specialized agents)
- Clear boundaries (client doesn't need to know server internals)

## 🔧 Customization

### Change Server URL

Edit `app/a2a_tool.py`:

```python
base_url = 'http://your-server:port'
```

### Add System Instructions

Modify `app/client_agent.py`:

```python
model = ChatOpenAI(
    model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    temperature=0,
).bind(
    system="You are a helpful assistant that delegates complex queries to a specialized A2A agent."
)
```

### Use Different LLM

Set in `.env`:

```bash
TOOL_LLM_NAME=gpt-4o  # or any other model
```

## 🐛 Troubleshooting

### "Could not connect to A2A server"

**Solution**: Make sure the server is running:
```bash
uv run python -m app
```

### "Request timed out"

**Cause**: Server is processing a complex query (helpfulness evaluation can take time)

**Solution**: The timeout is set to 60 seconds. For very complex queries, you may need to increase it in `a2a_tool.py`.

### "No response received"

**Cause**: Server returned an unexpected response format

**Solution**: Check server logs for errors. Ensure server is running the latest code.

## 📊 Expected Output

When running tests, you should see:

```
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
CLIENT AGENT TEST SUITE
Testing LangGraph Client Agent → A2A Protocol → Server Agent
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀

================================================================================
TEST 1: Web Search (Tavily Tool)
================================================================================

============================================================
Query: What are the latest developments in artificial intelligence in 2025?
============================================================

🔧 Calling A2A server agent...

📥 Response from A2A server:

[Server's response with web search results]

✅ Final Response:
[Client agent's final response]
```

## 🎓 Learning Outcomes

After working with this client agent, you should understand:

1. **A2A Protocol**: How agents communicate via standardized protocol
2. **Agent Composition**: Building complex systems from simple agents
3. **Tool Abstraction**: Treating entire agents as tools
4. **LangGraph Patterns**: Building agents with tool-calling capabilities
5. **Multi-Turn Conversations**: Context preservation across interactions

## 🚢 Next Steps

- Modify the client agent to add additional tools alongside the A2A tool
- Create multiple specialized server agents and have the client route to them
- Implement authentication for the A2A protocol
- Add error handling and retry logic
- Create a web UI for the client agent

---

This client agent demonstrates the power of A2A: **specialized agents working together through standardized protocols**! 🎉
