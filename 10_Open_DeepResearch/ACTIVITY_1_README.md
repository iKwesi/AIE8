# Activity 1: Configuration Experiments

This directory contains the code for Activity 1, which experiments with different configurations of the Deep Research system.

## Files

- **activity_1_experiments.py** - Main Python script with all experiments
- **experiment_results.json** - Generated results file (created after running)
- **ACTIVITY_1_README.md** - This file

## What This Does

Runs 4 experiments with different configurations and compares them using:
1. **Quantitative Metrics** (automatic tracking):
   - Execution time
   - Report length
   - Number of sources
   - Supervisor iterations
   - Researchers spawned

2. **LLM-as-a-Judge Evaluation** (single API call):
   - Comparative ranking of all reports
   - Qualitative assessment
   - Recommendations for different scenarios

## Experiments

1. **Increased Parallelism** - `max_concurrent_research_units: 10`
2. **Deeper Research** - `max_researcher_iterations: 8, max_react_tool_calls: 15`
3. **Anthropic Native Search** - `search_api: "anthropic"`
4. **Disabled Clarification** - `allow_clarification: False`

## Running the Script

### Prerequisites
```bash
# Ensure you have the required packages
pip install PyPDF2 langchain-anthropic langgraph
```

### Run All Experiments
```bash
python activity_1_experiments.py
```

This will:
1. Load the PDF document
2. Run all 4 experiments sequentially
3. Display quantitative metrics
4. Run LLM comparative evaluation
5. Save results to `experiment_results.json`

**Estimated time:** 5-10 minutes  
**API calls:** 4 research runs + 1 evaluation = 5 total

## Converting to Jupyter Notebook

To create a notebook from this Python file:

### Method 1: Manual Conversion

1. Create a new Jupyter notebook
2. Copy each section (marked with `# ===...===`) into separate cells:
   - Section 1 → Cell 1 (Imports)
   - Section 2 → Cell 2 (Helper Functions)
   - Section 3 → Cell 3 (Load PDF)
   - Section 4 → Cell 4 (Baseline Reference)
   - Section 5 → Cell 5 (Experiment 1)
   - Section 6 → Cell 6 (Experiment 2)
   - Section 7 → Cell 7 (Experiment 3)
   - Section 8 → Cell 8 (Experiment 4)
   - Section 9 → Cell 9 (Main Execution - see below)

3. **For Section 9 (Main Execution)**, adapt for notebook:

```python
# ============================================================================
# SECTION 9: RUN ALL EXPERIMENTS
# ============================================================================

# Run all experiments
results = {}

print("🔬 Running Experiment 1: Increased Parallelism...")
results['exp1'] = await run_experiment(config_exp1, "Experiment 1: Increased Parallelism")

print("🔬 Running Experiment 2: Deeper Research...")
results['exp2'] = await run_experiment(config_exp2, "Experiment 2: Deeper Research")

print("🔬 Running Experiment 3: Anthropic Native Search...")
results['exp3'] = await run_experiment(config_exp3, "Experiment 3: Anthropic Native Search")

print("🔬 Running Experiment 4: Disabled Clarification...")
results['exp4'] = await run_experiment(config_exp4, "Experiment 4: Disabled Clarification")

# Display metrics
display_metrics_table(results)

# Run LLM evaluation
evaluation = await evaluate_all_reports(results, baseline_report)
display_rankings(evaluation)
```

**Note:** Remove `if __name__ == "__main__":` and `asyncio.run(main())` - Jupyter handles async natively.

### Method 2: Using jupytext (Automated)

```bash
# Install jupytext
pip install jupytext

# Convert Python file to notebook
jupytext --to notebook activity_1_experiments.py

# This creates activity_1_experiments.ipynb
```

## Running Individual Experiments in Notebook

In a notebook, you can run experiments individually:

```python
# Run just Experiment 1
result_exp1 = await run_experiment(config_exp1, "Experiment 1")
print(result_exp1['final_report'])
```

Or run them all at once using the adapted Section 9 code above.

## Baseline Report

Before running the LLM evaluation, you need to add the baseline report from the main notebook:

1. Run the main notebook experiment
2. Copy the final report output
3. Replace the placeholder in Section 4:

```python
baseline_report = """
[PASTE YOUR BASELINE REPORT HERE]
"""
```

Alternatively, save the baseline report to a file and load it:

```python
with open("baseline_report.txt", "r") as f:
    baseline_report = f.read()
```

## Understanding the Output

### Quantitative Metrics Table
```
Metric                         Exp 1        Exp 2        Exp 3        Exp 4       
--------------------------------------------------------------------------------
Execution Time (s)             45.23        78.91        52.34        38.12       
Report Length (chars)          3200         4500         3100         2800        
Number of Sources              15           22           14           12          
Supervisor Iterations          2            8            2            2           
Researchers Spawned            10           1            1            1           
```

### LLM Evaluation
- Rankings (1st to 5th place)
- Justifications for each ranking
- Overall winner and why
- Insights about which config works best

## Customizing Experiments

To add your own experiment:

1. Create a new config (copy and modify existing):
```python
config_exp5 = {
    "configurable": {
        # Your custom settings here
        "max_concurrent_research_units": 5,
        # ...
    }
}
```

2. Run it:
```python
results['exp5'] = await run_experiment(config_exp5, "Experiment 5: My Custom Config")
```

3. Update the metrics table and evaluation to include exp5

## Troubleshooting

### API Key Issues
```python
# Set keys manually if getpass doesn't work
os.environ["ANTHROPIC_API_KEY"] = "your-key-here"
os.environ["TAVILY_API_KEY"] = "your-key-here"
```

### Import Errors
```bash
# Ensure open_deep_library is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/10_Open_DeepResearch"
```

### Memory Issues
If running all experiments causes memory issues, run them separately:
```python
# Run one at a time
result1 = await run_experiment(config_exp1, "Exp 1")
# Clear memory if needed
import gc; gc.collect()
result2 = await run_experiment(config_exp2, "Exp 2")
```

## Next Steps

1. ✅ Run the experiments
2. ✅ Analyze the metrics and rankings
3. ✅ Identify the best configuration for your use case
4. ✅ Try additional custom configurations
5. ✅ Document your findings

## Questions Answered

This activity helps answer:
- Which configuration is fastest?
- Which produces the most comprehensive reports?
- How does parallelism affect quality vs speed?
- Is deeper research worth the extra time/cost?
- How does search API choice impact results?
- What's the impact of skipping clarification?
