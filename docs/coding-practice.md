# Coding ownership and interview practice

The goal is not to memorise framework syntax. The goal is to explain and safely change every
important boundary in this repository.

## Weekly blank-file drills

1. Write a tiny PyTorch classifier loop: batch → logits → cross-entropy → zero grad → backward →
   optimiser step. Print the shapes at each boundary.
2. Write a three-node LangGraph: validate → decide → render. Add an `interrupt()` on one branch and
   resume it with an in-memory checkpointer.
3. Rebuild a city-first retriever with a two-record fixture and prove the other city's record cannot
   appear.

Documentation and autocomplete are allowed. Afterward, explain the code without looking at it.

## Three seeded debugging exercises

Work in a temporary branch and introduce each fault yourself; do not commit the broken state.

### 1. Tensor shape mismatch

Remove `.unsqueeze(0)` in `VisionClassifier.predict`. Observe the model error, inspect the tensor
shape, restore the batch dimension and add a shape assertion. Be ready to explain CHW versus NCHW.

### 2. Incorrect graph transition

Route the uncertain branch directly to `retrieve_rules`. Run the low-confidence workflow test,
explain why `confirmed_class` is missing, then restore `confirm_prediction` and resume semantics.

### 3. Wrong-city retrieval

Build one global FAISS candidate list and remove the pre-filter. Add a deliberately similar Zurich
and Gurugram query, observe the leak, then restore the per-city index. Explain why post-filtering is
weaker: the top-k results may already have excluded the right city's record.

## Questions you should answer confidently

- Why raw logits—not softmax probabilities—go into cross-entropy.
- Why macro-F1 is the early-stopping metric and why the test set remains untouched.
- What exactly triggers human confirmation and how LangGraph resumes the node.
- Where city filtering occurs, and how tests enforce it.
- Which fields the local LLM can influence (only `explanation`) and which it cannot.
- What happens if model weights, FAISS, Qwen or a specific rule is unavailable.
- Why this architecture is safer than asking one large model to identify and dispose of an item.

