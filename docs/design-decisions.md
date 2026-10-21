# Design decisions

## Tensor path

The input photo is converted to RGB, resized to 256 on its shorter side, center-cropped to
`224 × 224`, converted to `float32`, and normalized with ImageNet statistics. A single inference
batch therefore has shape `[1, 3, 224, 224]`. MobileNetV3-Small returns `[1, 9]` logits. Softmax is
used only for display/routing; weighted cross-entropy consumes raw logits during training.

Moderate crop, horizontal flip, 12-degree rotation and colour jitter apply only to training images.
Validation and test transforms are deterministic.

## Loss and evaluation

RealWaste classes are not perfectly balanced. Each class receives weight
`N / (number_of_classes × class_count)` in cross-entropy. Model selection uses validation macro-F1
so a large class cannot hide weak minority-class performance. The untouched test set is evaluated
once after selection. Accuracy, macro-F1, top-three accuracy, per-class precision/recall and the
confusion matrix are saved.

The split manifest contains absolute image paths and a single split label. Seed 42 and stratified
70/15/15 splitting make it reproducible. Tests assert that paths are unique.

## Graph state and transitions

```text
validate_input → classify_image → confidence_router
                                  ├─ uncertain → confirm_prediction (interrupt)
                                  └─ confident ───────────────────────────────┐
                                                                             ↓
retrieve_rules → generate_explanation → validate_grounding → render_result
```

Uncertainty means top confidence `< 0.70` **or** the top-two margin `< 0.15`. The checkpoint key is
the Gradio session/thread UUID. `interrupt()` sends only the top-three labels/confidences. Resume
accepts one supported material label, which becomes the confirmed class. A user correction follows
the same downstream route with `forced_class`; the trace marks it `user_override`.

Images are serialized temporarily as PNG bytes because checkpoints require serializable state.
The checkpointer is process memory only. Trace events contain node, elapsed milliseconds and safe
metadata—never image bytes or the free-text note.

## Retrieval boundary

The retriever constructs a separate FAISS index for each city. It selects the city's candidate set
before embedding similarity search, so a Zurich query cannot return a Gurugram RuleCard. The exact
confirmed class is promoted to rank one; similarity helps surface related rules but cannot replace
the class-specific source of truth.

If embeddings or FAISS cannot load, exact structured lookup still works. If retrieval is empty, the
graph creates a minimal source-only fallback that tells the user to open the municipal guide. It
does not invent a disposal route.

## Why the LLM cannot choose the bin

Municipal actions, preparation steps, exceptions, URLs and dates come directly from a validated
RuleCard. Qwen receives those facts only to write one short environmental-benefit sentence, with a
schema of one field. Parsing is retried once. Any import, load, parse or generation failure uses a
reviewed language-specific sentence.

After generation, `validate_grounding` compares action, steps, exceptions and source URL with the
retrieved RuleCard. The LLM has no interface through which it can change those fields. This is
safer and easier to test than asking a generative model to reproduce changing municipal rules.

## Limitations

- A nine-class image model cannot identify chemical, electrical or medical risk reliably.
- City-wide guidance may not capture building, ward or service-provider differences.
- Machine-assisted German/Hindi translations are convenience text, not legal authority.
- RealWaste may not represent local packaging, lighting and camera conditions equally.
- The in-memory checkpointer does not preserve an interrupted session after a Space restart.

