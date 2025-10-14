# Activity 1: Configuration Experiments - Summary

## What We Built

A complete experimental framework for comparing different Deep Research system configurations using a **hybrid evaluation approach**.

## Files Created

1. **activity_1_experiments.py** (520 lines)
   - Complete Python script with all experiments
   - Can run as standalone script or convert to notebook
   - Includes all helper functions and configurations

2. **ACTIVITY_1_README.md**
   - Comprehensive usage instructions
   - Notebook conversion guide
   - Troubleshooting tips

3. **ACTIVITY_1_SUMMARY.md** (this file)
   - Overview of what was built
   - Key features and design decisions

## Architecture

### Hybrid Evaluation Approach

**Quantitative Metrics (Automatic - No API Cost)**
- Execution time
- Report length (characters)
- Number of sources cited
- Supervisor iterations
- Researchers spawned

**Qualitative Evaluation (1 API Call)**
- LLM-as-a-judge comparative ranking
- Evaluates: comprehensiveness, accuracy, clarity, depth, source quality
- Provides justifications and recommendations

### Why Hybrid?

✅ **Cost-effective**: 1 LLM call vs 4 (if we evaluated each separately)  
✅ **Comprehensive**: Both quantitative and qualitative insights  
✅ **Comparative**: Directly ranks experiments against each other  
✅ **Practical**: Easy to see trade-offs (speed vs quality vs cost)

## Experiments Included

### Experiment 1: Increased Parallelism
- **Config**: `max_concurrent_research_units: 10`
- **Hypothesis**: More parallel researchers = faster execution, broader coverage
- **Expected**: Faster but potentially more expensive

### Experiment 2: Deeper Research
- **Config**: `max_researcher_iterations: 8, max_react_tool_calls: 15`
- **Hypothesis**: More iterations = deeper insights, more comprehensive
- **Expected**: Slower but higher quality

### Experiment 3: Anthropic Native Search
- **Config**: `search_api: "anthropic"`
- **Hypothesis**: Native search may provide better quality
- **Expected**: Different search results, possibly better integration

### Experiment 4: Disabled Clarification
- **Config**: `allow_clarification: False`
- **Hypothesis**: Skipping clarification speeds up workflow
- **Expected**: Faster but may miss important context

## Key Design Decisions

### 1. Skip Baseline Re-run
- Baseline already run in main notebook
- Saves API costs and time
- Referenced in LLM evaluation for comparison

### 2. Section-Based Structure
- Each section clearly marked with `# ===...===`
- Easy to copy to notebook cells
- Logical progression from setup to execution

### 3. Metrics Tracking
- Automatic collection during execution
- No manual intervention needed
- Formatted table output for easy comparison

### 4. Single LLM Evaluation Call
- All reports evaluated together
- Comparative ranking (1st to 5th)
- Insights about which config works best for what scenarios

### 5. Results Persistence
- Saves to `experiment_results.json`
- Can re-run evaluation without re-running experiments
- Useful for documentation and analysis

## How to Use

### As Python Script
```bash
python3 activity_1_experiments.py
```

### As Jupyter Notebook
1. Copy sections to notebook cells
2. Adapt Section 9 (remove `if __name__` and `asyncio.run`)
3. Run cells sequentially or individually

### Individual Experiments
```python
# Run just one experiment
result = await run_experiment(config_exp1, "Experiment 1")
```

## Expected Output

### Console Output
```
============================================================
STARTING ACTIVITY 1 EXPERIMENTS
============================================================

🔬 Running Experiment 1: Increased Parallelism...
✓ Experiment 1 completed in 45.23s
  - Report length: 3200 characters
  - Sources found: 15
  - Supervisor iterations: 2
  - Researchers spawned: 10

[... similar for other experiments ...]

============================================================
QUANTITATIVE METRICS COMPARISON
============================================================

Metric                         Exp 1        Exp 2        Exp 3        Exp 4       
--------------------------------------------------------------------------------
Execution Time (s)             45.23        78.91        52.34        38.12       
[... more metrics ...]

============================================================
LLM COMPARATIVE EVALUATION
============================================================

[Claude's comparative ranking and analysis]

============================================================
ALL EXPERIMENTS COMPLETE!
============================================================
```

### Generated Files
- `experiment_results.json` - Metrics and report previews

## Questions This Answers

From the notebook's Activity #1 section:

✅ **Which configuration is fastest?**  
→ Check execution time metrics

✅ **Which produces the most comprehensive reports?**  
→ Check report length, sources, and LLM evaluation

✅ **How does parallelism affect quality vs speed?**  
→ Compare Exp 1 (parallel) vs baseline

✅ **Is deeper research worth the extra time/cost?**  
→ Compare Exp 2 (deep) metrics and quality ranking

✅ **How does search API choice impact results?**  
→ Compare Exp 3 (Anthropic) vs baseline (Tavily)

✅ **What's the impact of skipping clarification?**  
→ Compare Exp 4 (no clarification) vs baseline

## Extensibility

Easy to add more experiments:

```python
# Add Experiment 5
config_exp5 = {
    "configurable": {
        "max_concurrent_research_units": 5,
        "max_researcher_iterations": 4,
        # ... custom settings
    }
}

results['exp5'] = await run_experiment(config_exp5, "Experiment 5")
```

## Next Steps for User

1. ✅ **Run the baseline** in main notebook (if not done)
2. ✅ **Copy baseline report** to Section 4 placeholder
3. ✅ **Run experiments** using the Python script
4. ✅ **Analyze results** from metrics and LLM evaluation
5. ✅ **Convert to notebook** for interactive exploration
6. ✅ **Document findings** in notebook markdown cells
7. ✅ **Try custom configurations** based on insights

## Technical Notes

### Syntax Validation
✅ File compiles successfully with `python3 -m py_compile`

### Dependencies
- PyPDF2 (PDF loading)
- langchain-anthropic (LLM calls)
- langgraph (research graph)
- open_deep_library (project library)

### Async Handling
- Script: Uses `asyncio.run(main())`
- Notebook: Uses top-level `await` (Jupyter native)

### Error Handling
- Graceful failure if report generation fails
- Metrics still collected even on errors
- Results saved even if some experiments fail

## Comparison: Full LLM Judge vs Hybrid

| Aspect | Full LLM Judge | Hybrid (Our Choice) |
|--------|---------------|---------------------|
| API Calls | 4 (one per exp) | 1 (comparative) |
| Cost | Higher | Lower |
| Detail | Very detailed per report | High-level comparison |
| Metrics | Only qualitative | Quant + Qual |
| Speed | Slower | Faster |
| Best For | Deep analysis | Quick comparison |

We chose **Hybrid** because it's more cost-effective and provides both quantitative data and qualitative insights.

## Success Criteria

✅ All experiments run successfully  
✅ Metrics collected automatically  
✅ LLM evaluation provides clear rankings  
✅ Results saved for later analysis  
✅ Easy to convert to notebook  
✅ Extensible for custom experiments  

## Conclusion

This framework provides a complete solution for Activity 1, allowing systematic comparison of different Deep Research configurations with both quantitative and qualitative evaluation. The hybrid approach balances cost, speed, and insight quality.
