<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 14: Build & Serve Agentic Graphs with LangGraph</h1>

| 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 14: Deploying Agents to Production](https://www.notion.so/Session-14-Deploying-Agents-to-Production-26acd547af3d80a59047c1685ff6d61a) |[Recording!](https://us02web.zoom.us/rec/share/P6sJWRwsWWf2cF91MXOzrlM40Tay-CqoLp5drxoS6AGQEvMD3krhLzGFcrhyuAh3.HWnYPtpB0DL2mrj2) (cQ2$d7E5) | [Session 14 Slides](https://www.canva.com/design/DAG2pZbibmw/YJHR3HSgG992FE1I-Mmwjw/edit?utm_content=DAG2pZbibmw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 14 Assignment: LangGraph_Platform](https://github.com/AI-Maker-Space/AIE8/tree/main/14_LangGraph_Platform) | [AIE8 Feedback 10/23](https://forms.gle/rSCtaKTaPkTeqoo1A)

# Build 🏗️

Run the repository and complete the following:

- 🤝 Breakout Room Part #1 — Building and serving your LangGraph Agent Graph
  - Task 1: Getting Dependencies & Environment
    - Configure `.env` (OpenAI, Tavily, optional LangSmith)
  - Task 2: Serve the Graph Locally
    - `uv run langgraph dev` (API on http://localhost:2024)
  - Task 3: Call the API from a different terminal
    - `uv run test_served_graph.py` (sync SDK example)
  - Task 4: Explore assistants (from `langgraph.json`)
    - `agent` → `simple_agent` (tool-using agent)
    - `agent_helpful` → `agent_with_helpfulness` (separate helpfulness node)

- 🤝 Breakout Room Part #2 — Using LangGraph Studio to visualize the graph
  - Task 1: Open Studio while the server is running
    - https://smith.langchain.com/studio?baseUrl=http://localhost:2024
  - Task 2: Visualize & Stream
    - Start a run and observe node-by-node updates
  - Task 3: Compare Flows
    - Contrast `agent` vs `agent_helpful` (tool calls vs helpfulness decision)

## Activities and Questions 🏗️ &❓

#### ❓ Question 1:

Compare the `agent` and `agent_helpful` assistants defined in `langgraph.json`. Where does the helpfulness evaluator fit in the graph, and under what condition should execution route back to the agent vs. terminate?

##### ✅ Answer:

## **Comparison: `agent` vs `agent_helpful`**

**`agent` (Simple Agent)**
- Basic ReAct loop: agent → tools → agent → END
- Terminates when no tool calls are needed

**`agent_helpful` (Agent with Helpfulness)**
- Enhanced loop: agent → tools → agent → **helpfulness evaluator** → decision
- Adds quality gate before termination

## **Quick Comparison Table**

| Aspect | agent (Simple) | agent_helpful (Enhanced) |
|:-------|:-----------------|:---------------------------|
| Quality Gate | None | Helpfulness evaluator |
| Self-Correction | No | Yes (up to 10 iterations) |
| Termination Logic | No tool calls → END | Helpfulness Y/N/Limit |
| Graph Flow | agent → tools → END | agent → tools → evaluator → decision |
| Cost Profile | Lower (single pass) | Higher (potential loops) |
| Use Case | Quick queries | Quality-critical tasks |



## **Helpfulness Evaluator Position**

The helpfulness node sits **between the agent's final response and termination**, acting as a quality checkpoint:

```
agent → [no tool_calls] → helpfulness → [Y/N/Limit] → end/continue
```

It only runs when the agent produces a final answer (no tool calls requested).


## **Routing Conditions**

### **Route Back to Agent (Continue)**
- **Trigger**: Helpfulness evaluation returns **'N'** (not helpful)
- **Action**: Loop back to agent for improved response
- **Purpose**: Iterative self-correction

### **Terminate (End)**
Three scenarios:
1. **Quality Met**: Evaluation returns **'Y'** (helpful)
2. **Loop Limit**: Message count exceeds **10** (safety mechanism)
3. **Force Stop**: Loop limit marker detected


## **Key Insight**

The `agent_helpful` assistant implements a **quality-assured feedback loop** where responses are automatically evaluated and improved until they meet quality standards or hit a safety limit. This prevents poor responses from reaching users while avoiding infinite loops through the 10-message cap.

**Trade-off**: Higher quality responses at the cost of additional API calls and latency.

#### 🏗️ Activity #1 Debugging A Graph

Select the `agent_with_helpfulness` and set one or more interrupts (at least one `Before` and one `After`). Try changing values and continuing the turn. 

#### ❓ Question 2:

What are your thoughts on when you would use a Before interrupt vs. an After interrupt?

##### ✅ Answer:

## **Before Interrupt vs After Interrupt: Strategic Use Cases**

Interrupts in LangGraph provide human-in-the-loop control at critical decision points. The key distinction lies in **timing and purpose**:

**Before interrupts** act as **proactive gatekeepers** - they pause execution before a node runs, allowing you to validate inputs, approve actions, or prevent potentially problematic operations. Think of them as "ask permission before acting."

**After interrupts** serve as **reactive quality controls** - they pause after a node completes, enabling you to review outputs, validate results, or modify state before continuing. Think of them as "trust but verify."

The strategic choice between them depends on whether you need **prevention** (before) or **validation** (after).

---

### **Quick Comparison Table**

| Aspect | **Before Interrupt** | **After Interrupt** |
|:-------|:---------------------|:--------------------|
| **Timing** | Pauses execution BEFORE node runs | Pauses execution AFTER node completes |
| **State Access** | Input state only | Input + output state |
| **Primary Use** | Prevention & validation | Review & modification |
| **Control Type** | Proactive (gate-keeping) | Reactive (quality control) |
| **Typical Action** | Approve/reject/modify inputs | Approve/reject/modify outputs |

---

## **When to Use Before Interrupts**

### **Use Cases:**

1. **Human-in-the-Loop Approval (Pre-execution)**
   - Approve expensive API calls before execution
   - Review tool parameters before invocation
   - Validate user intent before critical actions

2. **Input Validation & Sanitization**
   - Check if tool arguments are safe/valid
   - Verify cost thresholds before proceeding
   - Ensure compliance requirements are met

3. **Dynamic Routing Decisions**
   - Let humans choose between multiple paths
   - Override automatic routing logic
   - Inject additional context before processing

4. **Cost Control**
   - Prevent expensive operations (e.g., web scraping, API calls)
   - Budget enforcement checkpoints
   - Resource allocation decisions

### **Example Scenario:**
```
Before 'action' node (tool execution):
→ Pause to review: "About to search Arxiv for 'quantum computing' - Approve?"
→ Human can modify query or cancel
→ Prevents wasted API calls on poorly formed queries
```

---

## **When to Use After Interrupts**

### **Use Cases:**

1. **Output Quality Review**
   - Verify tool results before continuing
   - Check if retrieved information is relevant
   - Validate generated content meets standards

2. **Result Modification**
   - Edit/enhance tool outputs
   - Filter sensitive information
   - Augment results with additional context

3. **Decision Validation**
   - Review agent's reasoning before final response
   - Confirm helpfulness evaluation is accurate
   - Override automatic quality assessments

4. **Debugging & Observability**
   - Inspect intermediate states
   - Verify data transformations
   - Catch errors in processing logic

5. **Multi-Step Approval Workflows**
   - Review each stage of complex operations
   - Ensure progressive quality gates
   - Build audit trails

### **Example Scenario:**
```
After 'helpfulness' node:
→ Pause to review: "Helpfulness evaluator returned 'N' - Continue loop?"
→ Human can override decision or modify state
→ Prevents unnecessary re-generation if response is actually good
```

---

## **Strategic Decision Framework**

### **Choose Before Interrupt When:**
- ✅ You want to **prevent** potentially problematic actions
- ✅ Input validation is critical
- ✅ Cost/resource control is priority
- ✅ You need to **gate-keep** before execution
- ✅ Proactive human oversight required

### **Choose After Interrupt When:**
- ✅ You want to **review** completed work
- ✅ Output quality is the concern
- ✅ Results need human validation
- ✅ You need to **inspect** execution results
- ✅ Reactive quality control required

---

## **Production Patterns**

### **Before Interrupt Pattern: "Ask Before Acting"**
```python
# Interrupt before expensive tool execution
graph.add_node("action", tool_node, interrupt="before")
```
**Best for**: Financial transactions, data deletion, external API calls

### **After Interrupt Pattern: "Trust But Verify"**
```python
# Interrupt after generation to review output
graph.add_node("agent", call_model, interrupt="after")
```
**Best for**: Content moderation, compliance checks, quality assurance

### **Combined Pattern: "Full Control"**
```python
# Both before and after for critical nodes
graph.add_node("critical_action", sensitive_operation, 
               interrupt=["before", "after"])
```
**Best for**: High-stakes operations, regulated environments, learning systems

---

## **Key Insight**

**Before interrupts** are about **prevention and control** (stopping bad things from happening).

**After interrupts** are about **validation and correction** (fixing things that happened).

In production systems, use **before interrupts** for irreversible or expensive operations, and **after interrupts** for quality assurance and compliance verification. Combining both creates comprehensive human-in-the-loop workflows for mission-critical applications.



<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

- Create and deploy a locally hosted MCP server with FastMCP.
- Extend your tools in `tools.py` to allow your LangGraph to consume the MCP Server.
</details>

# Ship 🚢

- Running local server (`langgraph dev`)
- Short demo showing both assistants responding

# Share 🚀
- Walk through your graph in Studio
- Share 3 lessons learned and 3 lessons not learned

# Main Homework Assignment

Follow these steps to prepare and submit your homework assignment:
1. Create a branch of your `AIE8` repo to track your changes. Example command: `git checkout -b s14-assignment`
2. Complete the Tasks listed in the Breakout Room sections of `Build 🏗️`
3. Complete the activities and questions in `Activities and Questions 🏗️ &❓` by editing the file and replacing "_(enter answer here)_" with your responses
3. Commit, and push your completed notebook to your `origin` repository. _NOTE: Do not merge it into your main branch._
4. Record a Loom video reviewing the content of your completed notebook
5. Make sure to include all of the following on your Homework Submission Form:
    + The GitHub URL to the `README.md` file _on your assignment branch (not main)_
    + The URL to your Loom Video
    + Your Three Lessons Learned/Not Yet Learned
    + The URLs to any social media posts (LinkedIn, X, Discord, etc.) ⬅️ _easy Extra Credit points!_


### OPTIONAL: 🚧 Advanced Build Assignment 🚧
<details>
  <summary>(<i>Open this section for the submission instructions.</i>)</summary>

Follow these steps to prepare and submit your homework assignment:
1. Create a branch of your `AIE8` repo to track your changes. Example command: `git checkout -b s14-assignment`
2. Create your MCP server
3. Add it to the existing graph's tools
4. Deploy it ***locally***
5. Validate the graph uses the MCP server's tools
6. Commit, and push your changes to your `origin` repository. _NOTE: Do not merge it into your main branch._
7. Record a Loom video reviewing the content of your completed notebook.
8. Make sure to include all of the following on your Homework Submission Form:
    + The GitHub URL to the notebook you created for the Advanced Build Assignment _on your assignment branch_
    + The URL to your Loom Video
    + Your Three Lessons Learned/Not Yet Learned
    + The URLs to any social media posts (LinkedIn, X, Discord, etc.) ⬅️ _easy Extra Credit points!_

</details>
