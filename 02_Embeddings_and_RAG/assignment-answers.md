#### ❓Question #1:

The default embedding dimension of `text-embedding-3-small` is 1536, as noted above. 

### Q1. Is there any way to modify this dimension?
2. What technique does OpenAI use to achieve this?

> NOTE: Check out this [API documentation](https://platform.openai.com/docs/api-reference/embeddings/create) for the answer to question #1.1, and [this documentation](https://platform.openai.com/docs/guides/embeddings/use-cases) for an answer to question #1.2!

##### ✅ Answer:
1. Yes, yBy default, text-embedding-3-small produces embeddings of length 1536, but OpenAI allows you to shorten the vector using the dimensions parameter. You can set this parameter to any value between 1 and 1536, depending on how much compression you want. You cannot increase the embedding size beyond 1536 for this model.
Example in code:

```python
# Example using the dimensions parameter
response = client.embeddings.create(
    input="Your text here",
    model="text-embedding-3-small",
    dimensions=512  # Shortened from default 1536
)
```

## 2. What technique does OpenAI use to achieve this?

OpenAI uses a technique known as **Matryoshka Representation Learning** (named after the nested Russian dolls).  

### How it works:
- The embedding models are trained so that **information is hierarchically ordered across dimensions**.  
- The **earlier dimensions encode the most essential semantic information**, while later dimensions refine or add nuance.  
- Because of this ordering, you can truncate an embedding (or request a smaller `dimensions` value), and the smaller vector will still capture the core meaning of the text.  
- This makes embeddings of different sizes **nested and compatible** — a 512-dimensional vector is essentially a prefix of the 1536-dimensional one.  

This training approach ensures that embeddings of different lengths can still be used interchangeably in tasks like semantic search, clustering, or classification with only a modest trade-off in accuracy.  

---

#### ❓Question #2:

What are the benefits of using an `async` approach to collecting our embeddings?

> NOTE: Determining the core difference between `async` and `sync` will be useful! If you get stuck - ask ChatGPT!

### Core difference: `sync` vs `async`
- **Synchronous (`sync`)**: Each API call is made **one at a time**, and the program waits for the response before sending the next request.  
- **Asynchronous (`async`)**: Multiple API calls can be **in flight at once**, and the program doesn’t block while waiting. It handles responses as they arrive.

---
### Benefits of using `async`
1. **Faster throughput**  
   - You can send many embedding requests concurrently instead of waiting for each one to finish.  
   - This significantly reduces runtime when embedding large datasets.

   ```python
   # Sync approach - slow, sequential
    def sync_get_embeddings(texts):
        embeddings = []
        for text in texts:  # Each call blocks until complete
            embedding = openai_api_call(text)
            embeddings.append(embedding)
        return embeddings

    # Async approach - fast, concurrent
    async def async_get_embeddings(texts):
        tasks = [openai_api_call(text) for text in texts]
        return await asyncio.gather(*tasks)  # All calls happen simultaneously
   ```

2. **Better utilization of rate limits**  
   - OpenAI APIs enforce requests-per-minute and tokens-per-minute limits.  
   - Async helps keep requests flowing smoothly, maximizing your quota usage.

3. **Improved scalability**  
   - Async avoids bottlenecks when embedding at scale (thousands or millions of texts).  
   - It makes pipelines involving embeddings (e.g., semantic search or classification) more efficient.

   ```python
    # Processing 1000 documents:
    # Sync: 1000 texts × 200ms per API call = 200 seconds
    # Async: 1000 texts ÷ concurrent_limit × 200ms = ~20 seconds (10x faster)

    texts = ["doc1", "doc2", ...] * 1000
    start = time.time()
    embeddings = await async_get_embeddings(texts)
    print(f"Processed {len(texts)} embeddings in {time.time() - start:.2f}s")
   ```

4. **Non-blocking execution**  
   - While waiting for embeddings, your program can do other work (e.g., logging, preprocessing, writing results).  
   - This leads to better overall resource usage.

---

### ✅ In short
Using an `async` approach makes embedding collection **faster, more efficient, and scalable** because you can issue multiple requests concurrently, avoid idle waiting, and fully utilize API rate limits.

## Question #3: 
When calling the OpenAI API — are there ways to achieve more reproducible outputs?

**Answer:**

Yes — there *are* ways to make outputs more reproducible, though perfect determinism is not always guaranteed. To increase reproducibility:

- Use the `seed` parameter (when supported): set it to a fixed integer across repeated calls.  
```python
    def run(self, messages, text_only: bool = True, seed: int = 42):
        response = openai.ChatCompletion.create(
        model=self.model_name, 
        messages=messages,
        temperature=0,
        seed=seed  # Enables reproducible outputs across calls
    )
    return response.choices[0].message.content if text_only else response
```
- Use identical request parameters each time (prompt wording, temperature, `top_p`, `max_tokens`, etc.).  
- Set randomness-controlling parameters to deterministic values — for example, `temperature = 0` (or very low), possibly a fixed `top_p`.  
```python
    def run(self, messages, text_only: bool = True):
        response = openai.ChatCompletion.create(
        model=self.model_name, 
        messages=messages,
        temperature=0  # Makes outputs deterministic
    )
    return response.choices[0].message.content if text_only else response
```
- Examine the `system_fingerprint` returned by the API: if this changes, the backend or model configuration has changed, which can cause output drift.  

**Notes / caveats:**

- Reproducibility settings are in preview / beta for certain models only.  
- Even with seed + identical parameters + same fingerprint, output may *still* vary in some cases due to backend changes, implementation details, or non-deterministic hardware/software behavior.  

---

#### ❓ Question #4:

What prompting strategies could you use to make the LLM have a more thoughtful, detailed response?

What is that strategy called?

> NOTE: You can look through our [OpenAI Responses API](https://colab.research.google.com/drive/14SCfRnp39N7aoOx8ZxadWb0hAqk4lQdL?usp=sharing) notebook for an answer to this question if you get stuck!

**Answer:**

One effective strategy is to explicitly prompt the model to **reason step by step** before giving its final answer.  
Examples include:  
- “Let’s think step by step.”  
- “Explain your reasoning in detail.”  
- “List the considerations before deciding.”  

This strategy encourages the model to break down its reasoning, explore multiple aspects, and produce a more thoughtful, detailed response.

**This is called _Chain-of-Thought (CoT) prompting_.**

Here are some variations of Chain of Thoughts:
---

### 1. Basic Chain-of-Thought (CoT)
Ask the model directly to think step by step.

```python
def run_with_cot(self, user_message: str, text_only: bool = True):
    """
    Basic structured reasoning:
    - Instructs the model to think step-by-step internally.
    - Outputs only a concise, structured rationale and final answer.
    - Encourages uncertainty handling and clear assumptions.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful reasoner. Think step by step internally (do not reveal your hidden chain-of-thought). "
                "Output a concise, structured response with the following sections:\n"
                "1) Assumptions (if any)\n"
                "2) Key Steps (3–5 bullets max)\n"
                "3) Final Answer\n"
                "If information is missing, clearly state what is needed."
            )
        },
        {
            "role": "user",
            "content": (
                f"{user_message}\n\n"
                "Provide your response using the exact headings:\n"
                "Assumptions:\n"
                "Key Steps:\n"
                "Final Answer:"
            )
        },
    ]
    return self.run(messages, text_only)
```
---

### 2. Enhanced CoT with Structured Thinking (Decompose → Analyze → Conclude)

```python
    def run_detailed_analysis(self, user_message: str, text_only: bool = True):
    """
    Enhanced structured thinking:
    - Enforces a clear scaffold for decomposition, multiple perspectives, trade-offs, and conclusions.
    - Asks for brief, decision-relevant rationale (not full chain-of-thought).
    - Includes edge cases and explicit risks/uncertainties.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a thorough analyst. Think step by step internally (do not reveal your hidden chain-of-thought). "
                "Return a structured analysis with these sections:\n"
                "1) Problem Breakdown\n"
                "2) Options / Perspectives\n"
                "3) Pros & Cons (concise bullets)\n"
                "4) Risks & Edge Cases\n"
                "5) Recommendation (with brief rationale)\n"
                "Keep each section focused and under 6 bullets. If data is missing, note the gaps."
            )
        },
        {"role": "user", "content": user_message},
    ]
    return self.run(messages, text_only)

```

### 3. Few-Shot CoT (Teach by Example, with Structured Outputs)

```python
    def run_with_examples(self, user_message: str, text_only: bool = True):
    """
    Few-shot structured examples:
    - Demonstrates the desired format and depth using short exemplars.
    - Models internal reasoning but outputs structured, concise rationale + answer.
    - Encourages portability across topics by focusing on reusable scaffolds.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Provide thoughtful, high-quality responses with structured brief rationale (no hidden chain-of-thought in the output). "
                "Use the headings: Context, Key Steps, Recommendations, Final Answer."
            )
        },

        # EXAMPLE 1
        {"role": "user", "content": "How do I optimize database queries?"},
        {
            "role": "assistant",
            "content": (
                "Context:\n"
                "- Workload has slow SELECTs with JOINs; mixed OLTP/analytics.\n\n"
                "Key Steps:\n"
                "- Examine slow query logs; identify high-latency statements.\n"
                "- Check execution plans; look for full scans and poor join orders.\n"
                "- Add/adjust composite indexes based on predicates and join keys.\n"
                "- Normalize hotspots or create selective materialized views.\n"
                "- Batch writes; separate OLTP from analytics if needed.\n\n"
                "Recommendations:\n"
                "- Create indexes on (customer_id, created_at) and on frequent join keys.\n"
                "- Rewrite correlated subqueries as joins; limit SELECT columns.\n"
                "- Add covering indexes for top 5 slow queries.\n\n"
                "Final Answer:\n"
                "Profile queries → inspect plans → fix access paths (indexes/rewrites) → validate with benchmarks."
            )
        },

        # EXAMPLE 2
        {"role": "user", "content": "Outline a plan to reduce mobile app crash rates."},
        {
            "role": "assistant",
            "content": (
                "Context:\n"
                "- Crash spikes on older Android devices after latest release.\n\n"
                "Key Steps:\n"
                "- Aggregate crash logs by device/OS/version; identify top offenders.\n"
                "- Reproduce critical crashes locally and in CI.\n"
                "- Add defensive checks around null / lifecycle / threading.\n"
                "- Implement feature flags and staged rollouts.\n\n"
                "Recommendations:\n"
                "- Add instrumentation to isolate ANRs and OOMs.\n"
                "- Ship hotfix for top 2 crashes; monitor with staged rollout (5% → 25% → 100%).\n\n"
                "Final Answer:\n"
                "Triage by impact, fix reproducible root causes, guard with flags, and roll out gradually with monitoring."
            )
        },

        # TARGET PROMPT
        {"role": "user", "content": user_message},
    ]
    return self.run(messages, text_only)

```
---

### 4. Role-Based Expert Prompting (+ Self-Check)

```python
    def run_as_expert(self, user_message: str, expertise_domain: str, text_only: bool = True):
    """
    Role-based expert prompt:
    - Adopts a senior expert persona for domain-grounded recommendations.
    - Adds a lightweight self-check pass to catch obvious mistakes.
    - Avoids revealing hidden chain-of-thought; outputs crisp reasoning + answer.
    """
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a senior {expertise_domain} expert with 15+ years of experience. "
                "Think step by step internally (do not reveal your hidden chain-of-thought). "
                "Respond with:\n"
                "1) Context & Assumptions\n"
                "2) Analysis (concise, decision-relevant)\n"
                "3) Recommendation (with brief rationale)\n"
                "4) Self-Check (list 2–3 quick validations or potential pitfalls)\n"
                "If uncertain, state what would increase confidence (data/tests)."
            )
        },
        {"role": "user", "content": user_message},
    ]
    return self.run(messages, text_only)

```
---

## 🏗️ Activity #1:
Enhance your RAG application in some way!

