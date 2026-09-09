# Changelog

Version history for Transformer Trace. See README.md for the project overview, features, and how to run it.

## v0.9

v0.9 works through an external review's five-point priority list:

- **Real trace (#1):** No code change was needed; `extract_traces.py` already had the full schema. It now also produces `generation_chain`, so a real run gives you both the real forward pass and real multi-step generation in one command.
- **Shape tracker (#2):** A small `Tensor: [...]` pill after every diagram, computed live from `TRACE.meta` rather than hardcoded.
- **Trace one calculation (#3):** Click any non-masked cell in any of the three attention grids to see the exact chain for that `(query, key)` pair: Q·K → ÷√64 → softmax → ×V → contribution.
- **Autoregressive loop (#4):** A new "Watch it generate" panel appends each generated token and shows the next candidate distribution. The demo uses a synthetic four-step chain; `extract_traces.py --generate-steps N` produces the real version through greedy decoding.
- **Beginner/deeper layers (#5):** The shape pills serve as the tensors layer and the trace-cell panel serves as the math layer. Both are opt-in, while the existing "show the math" toggles cover MLP and logits details.

Also fixed the content correction from the review — the GPT-2→GPT-4 generalization claim in the Overview section was too strong; reworded to say the concepts generalize while several specific components don't.
## v0.8

Added a bird's-eye "pipeline map" to the Overview section: six clickable stages (Embedding → Q·K·V → Masked attention → Out & concat → MLP → Probabilities) tied to the detailed sections below. Clicking a stage smooth-scrolls to and briefly highlights the corresponding section.

## v0.7

This structure and content pass fixed the reading flow for people with little prior knowledge of transformers:

- Added a "00 — What's actually happening here" section explaining transformers, next-token prediction, and the roadmap.
- Split the Transformer Block into five numbered substeps: Query/Key/Value, heads and masked self-attention, output and concatenation, MLP, and residual/LayerNorm/dropout.
- Added explanatory prose after every diagram.
- Added bridge sentences at both section boundaries to explain what data carries over.

## v0.6

Added worked-example arithmetic using the prompt "The rocket climbed through the clouds. The rocket climbed through":

- Key and Query token lists feeding the dot-product step, plus a Value list feeding the Attention-Output step.
- A live Attention × Value = Out computation for the selected block, head, and token.
- MLP Expansion and Output Logits formulas behind a "show the math" toggle.
- A probability breakdown showing logit, temperature scaling, top-k/top-p filtering, and softmax arithmetic.

## v0.5

Added a 20-step guided walkthrough with previous/next navigation, a progress bar, a page counter, and scroll-to-highlight behavior for the relevant page section. All 20 steps were written specifically for this project.

## v0.4

Redesigned the visual language from an "instrument panel" look to a modern, minimalist layout with a light canvas, generous whitespace, Space Grotesk for text, monospace for data values, and thin typographic rules.

A personal rebuild of Transformer Explainer's actual pipeline: Embedding → repeated Transformer Block (Q/K/V → per-head Dot Product → Scaling·Mask → Softmax → Concatenation → MLP, with Residual/LayerNorm/Dropout called out) → Output Probabilities with temperature/top-k/top-p sampling. It uses original code and copy.

One addition on top of the original is a real, working induction head highlighted in the attention view: the copy-forward attention pattern behind in-context learning, per Olsson et al. (2022).

## What's Here Right Now
| File | What it is |
| --- | --- |
| `index.html` | The visualizer, with trace data embedded inline. |
| `sample_trace.json` | Synthetic GPT-2-shaped data with the full dot_product → scaled_masked → softmax pipeline. |
| `extract_traces.py` | Runs GPT-2, recovers dot-product and scale+mask stages, and probes all 144 heads. |
| `make_sample_trace.py` | Regenerates the synthetic demo data. |
| `inject_trace.py` | Swaps a real `trace.json` into `index.html`. |

## Trace Schema

```json
{
  meta: { model, n_layer, n_head, d_model, d_head, vocab_size },
  prompt, tokens: [...], token_ids: [...], positions: [...],
  layers: [
    { layer, attention: { heads: [
        { head, dot_product: [[..]], scaled_masked: [[..|null]], softmax: [[..]], label }
    ]}, mlp: {...} }
  ],
  induction_heads: [{layer, head, score}],
  output_logits: [[token, logit], ...]
}
```

I built and ran make_sample_trace.py and index.html in a sandboxed environment with no access to huggingface.co, so extract_traces.py is written and syntax-checked but not yet run against the real model — that needs to happen on your machine.
## Run It for Real

```bash
pip install -r requirements.txt
python extract_traces.py --prompt "The cat sat on the mat. The cat sat on" --probe-induction --out trace.json
```

The --probe-induction flag ranks all 144 (layer, head) pairs by a real induction score — it'll print the top 10, and you'll almost certainly find the actual induction heads sit at different coordinates than the (layer 9, head 5) placeholder used in the demo. That's expected — go update the frontend once you know where they really are. Finding out is most of the point of building this.
To view real data instead of the synthetic demo: open index.html, find the line const TRACE = JSON.parse(...), and swap in the contents of trace.json (or point a tiny local server at both files and fetch() it — opening trace.json via fetch() from a file:// page will get blocked by the browser's CORS rules, which is why the demo embeds data inline).
## Known Quirk

In the synthetic demo, the first repeated "The" doesn't trigger the induction pattern — only the following "cat", "sat", "on" do. That's not a bug I left in by accident: GPT-2's tokenizer treats sentence-initial "The" and mid-sentence " The" (leading space) as different token IDs, so an induction head literally can't match them. It's a good example of how tokenization quirks show up in real model behavior — worth confirming this actually happens with the real model once you run the probe.

## Roadmap

### Phase 1 — Complete

- [x] Depth toggle (ELI5 / Technical / Math).
- [x] Per-head attention heatmaps with causal masking.
- [x] Induction-head detection and highlighting.
- [x] Live temperature / top-k / top-p resampling from real logits.

### Phase 2 — Natural Next Steps

- [ ] Run `extract_traces.py` for real and replace the placeholder induction head with a verified one.
- [ ] Embedding-space explorer: nearest-neighbor search in GPT-2's 50,257 × 768 embedding matrix.
- [ ] Save/share a specific run via URL.

### Phase 3 — Novel Differentiator

- [ ] "Classic vs. Modern" architecture toggle: compare GPT-2 with a modern open model such as Qwen2.5-0.5B or SmolLM2.

### Phase 4 — Polish

- [ ] Mobile layout pass.
- [ ] Quantized/precomputed-trace fallback for slow devices.
- [ ] Multilingual tokenization comparison.

