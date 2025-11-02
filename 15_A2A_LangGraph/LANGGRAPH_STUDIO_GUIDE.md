# 🎨 LangGraph Studio Guide

This guide explains how to visualize and interact with your agents using LangGraph Studio.

## 🚀 Quick Start

### Start LangGraph Studio

```bash
uv run langgraph dev --allow-blocking
```

**Note:** The `--allow-blocking` flag is required because some package imports (like `langchain_tavily`) perform synchronous I/O operations when loading metadata. This is safe for local development and doesn't affect the async execution of your agents.

This will start the LangGraph development server and open LangGraph Studio in your browser.

**Default URLs:**
- **Studio UI**: https://smith.langchain.com/studio (with local connection)
- **API Server**: http://localhost:2024

## 📊 Available Agents

The `langgraph.json` configuration exposes two agents:

### 1. **server_agent** - A2A Server with Helpfulness Loop

**What it does:**
- Main agent with Tavily web search, ArXiv papers, and RAG retrieval
- Includes helpfulness evaluation loop (up to 10 iterations)
- This is the agent that runs at `http://localhost:10000` when you do `uv run python -m app`

**Graph Structure:**
```
agent → (tool calls?) → action → agent
     ↓ (no tools)
helpfulness → (helpful?) → END
            ↓ (not helpful, < 10 loops)
          agent (retry)
```

**Best for:**
- Understanding the helpfulness evaluation loop
- Seeing how tools are selected and executed
- Debugging tool interactions
- Visualizing the A2A evaluation process

### 2. **client_agent** - Simple A2A Client

**What it does:**
- Simple agent with ONE tool: `call_a2a_agent`
- Delegates all work to the server agent via A2A protocol
- Demonstrates agent-to-agent communication

**Graph Structure:**
```
agent → (needs A2A call?) → action (call_a2a_agent) → agent → END
```

**Important:** For this agent to work in Studio, you need the A2A server running:
```bash
# In a separate terminal
uv run python -m app
```

**Best for:**
- Understanding A2A protocol flow
- Seeing how one agent calls another
- Testing client-server architecture
- Visualizing tool abstraction

## 🎯 How to Use LangGraph Studio

### 1. Select an Agent

In the Studio UI, you'll see a dropdown to select between:
- `server_agent`
- `client_agent`

### 2. Configure Input

In the input panel, provide a message. **Note:** LangGraph Studio automatically wraps your input in the state structure, so you only need to provide the message object itself, not wrapped in a `messages` array.

**For server_agent:**
```json
{
  "role": "user",
  "content": "What are the latest AI developments in 2025?"
}
```

**For client_agent:**
```json
{
  "role": "user", 
  "content": "Find recent papers on transformers"
}
```

### 3. Run and Visualize

Click "Run" to execute the agent. You'll see:
- **Graph visualization**: Nodes and edges showing execution flow
- **State updates**: How the state changes at each step
- **Tool calls**: What tools are invoked and their results
- **Messages**: The conversation history

### 4. Step Through Execution

Use the step controls to:
- **Step forward**: Execute one node at a time
- **Step back**: Review previous states
- **Replay**: Re-run the entire execution

## 🔍 What to Look For

### In server_agent:

1. **Tool Selection**: Watch how the agent decides which tool to use (Tavily, ArXiv, or RAG)
2. **Helpfulness Loop**: See the evaluation process after each response
3. **Iteration Count**: Track how many times it loops before getting a helpful response
4. **State Evolution**: Observe how messages accumulate in the state

### In client_agent:

1. **A2A Tool Call**: See when the agent decides to call the A2A server
2. **Protocol Flow**: Watch the HTTP request/response via A2A protocol
3. **Response Handling**: How the client processes the server's response
4. **Simplicity**: Notice how the client doesn't need to know about Tavily/ArXiv/RAG

## 🧪 Test Scenarios

### Scenario 1: Web Search (server_agent)
```json
{
  "role": "user",
  "content": "What's happening in AI today?"
}
```
**Expected:** Uses Tavily tool, evaluates helpfulness, returns web results

### Scenario 2: Academic Papers (server_agent)
```json
{
  "role": "user",
  "content": "Find papers on attention mechanisms"
}
```
**Expected:** Uses ArXiv tool, retrieves papers, evaluates helpfulness

### Scenario 3: Document Retrieval (server_agent)
```json
{
  "role": "user",
  "content": "What do the documents say about AI usage?"
}
```
**Expected:** Uses RAG tool, retrieves from local PDFs, evaluates helpfulness

### Scenario 4: A2A Communication (client_agent)
**Prerequisites:** Server must be running (`uv run python -m app`)

```json
{
  "role": "user",
  "content": "Search for recent AI news"
}
```
**Expected:** Client calls A2A tool → Server processes → Client returns result

### Scenario 5: Multi-Turn Conversation
For multi-turn conversations, use the conversation history in Studio's interface. After the first query completes, simply add a follow-up message in the same thread.

**Turn 1:**
```json
{
  "role": "user",
  "content": "Find papers on LLMs"
}
```

**Turn 2:** (in the same thread, just add another message)
```json
{
  "role": "user",
  "content": "Summarize the key findings"
}
```

## 🐛 Troubleshooting

### "Graph not found" error

**Cause:** LangGraph can't import the graph function

**Solution:** 
1. Check that the path in `langgraph.json` is correct
2. Ensure the function is exported (not private with `_`)
3. Restart `langgraph dev`

### Client agent fails with connection error

**Cause:** A2A server is not running

**Solution:**
```bash
# Start the server in a separate terminal
uv run python -m app
```

### "Module not found" errors

**Cause:** Dependencies not installed

**Solution:**
```bash
uv sync
```

### Slow execution

**Cause:** Helpfulness evaluation can take 10-30 seconds

**Solution:** This is normal - the server agent evaluates each response for quality

## 📝 Adding More Agents

When you create additional agents (like for the Advanced Build), add them to `langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "server_agent": "./app/agent_graph_with_helpfulness.py:build_agent_graph_with_helpfulness",
    "client_agent": "./app/client_agent.py:build_client_agent",
    "persona_agent": "./app/persona_agent.py:build_persona_agent"
  },
  "env": ".env"
}
```

Then restart `langgraph dev` to see the new agent in the dropdown.

## 🎓 Learning Tips

1. **Start with server_agent**: Understand the core functionality first
2. **Watch the helpfulness loop**: See how A2A evaluation works
3. **Compare with client_agent**: Understand the abstraction layer
4. **Experiment with queries**: Try different types to trigger different tools
5. **Use step-through**: Don't just run - step through to understand each decision

## 🔗 Useful Links

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Studio Docs](https://langchain-ai.github.io/langgraph/cloud/deployment/studio/)
- [A2A Protocol Spec](https://github.com/google/a2a)

---

Happy visualizing! 🎨✨
