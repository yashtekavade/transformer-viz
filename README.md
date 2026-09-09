# Transformer Trace

A step-by-step, click-into-the-math trace of one GPT-2 forward pass — built from scratch as a deeper, more interactive take on Transformer Explainer.
<!-- Add a screenshot or short GIF of the page here, e.g.: --> <!-- ![Transformer Trace screenshot](docs/screenshot.png) -->
[Live demo](https://transformer-vizz.vercel.app/).

## What This Is

Most transformer explainers show you that attention happens. This one is built around also showing why the numbers are what they are — every heatmap has real arithmetic behind it, one click away.

Pick a prompt, pick a block, pick a head, and:

- Watch the actual dot product → ÷√64 → mask → softmax chain for any single `(query, key)` pair.
- See the real Attention × Value = Out sum for any token, live.
- Expand the MLP and logits projections into Embeddings × Weights + Bias = Output, with numbers plugged in.
- Click a probability bar and get the exact softmax arithmetic that produced its percentage.
- Generate real text one token at a time and watch the loop repeat.

All of it ships with synthetic-but-structurally-real data by default (correct shapes, causal masking, row-stochastic attention — just not real learned weights), so it works instantly with zero setup. A companion script swaps in a real GPT-2 forward pass — including real multi-step generation — when you want it.

## Features

- **Guided walkthrough:** 20-step narrated tour with a progress bar that scrolls to and highlights the relevant part of the page.
- **Pipeline map:** A six-node clickable overview of the whole flow: Embedding → Q·K·V → Masked attention → Out & concat → MLP → Probabilities.
- **Trace one calculation:** Click any cell in the attention grids to see that pair's full arithmetic chain.
- **Shape tracker:** A live `Tensor: [seq, dim]` pill after every stage, computed from the actual prompt length.
- **Worked formulas:** MLP expansion and output logits shown symbolically and with real numbers behind a "show the math" toggle.
- **Probability breakdown:** Logit → temperature-scaled logit → top-k/top-p filter → softmax for any token you click.
- **Generation loop:** Click through real autoregressive next-token generation, one token at a time.
- **Induction head:** One attention head is marked as behaving like a real induction head, the copy-forward mechanism behind in-context learning.

## Quick Start

### View the Demo (Zero Setup)

Clone the repo and open index.html directly in a browser — the trace data is embedded inline, no server or build step needed.
```bash
git clone https://github.com/yashtekavade/transformer-viz.git
cd transformer-viz
open index.html   # or just double-click it
```

### Run It Against Real GPT-2

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt

python extract_traces.py \
  --prompt "The rocket climbed through the clouds. The rocket climbed through" \
  --probe-induction \
  --generate-steps 6 \
  --out trace.json

python inject_trace.py trace.json --out index_real.html
```

Open `index_real.html`. The first run downloads GPT-2 small (~500 MB) from Hugging Face, so it needs internet access. `--probe-induction` ranks all 144 attention heads by a real induction score; `--generate-steps` controls how many real tokens the generation-loop panel gets to play with.

## Project Structure

| File | Purpose |
| --- | --- |
| `index.html` | The self-contained visualizer. |
| `sample_trace.json` | Synthetic demo data already embedded in `index.html`. |
| `extract_traces.py` | Real trace extraction for attention, MLP, logits, and generation. |
| `make_sample_trace.py` | Regenerates `sample_trace.json`. |
| `inject_trace.py` | Swaps a `trace.json` into `index.html`. |
| `requirements.txt` | Python dependencies. |
| `README.md` | Project overview and usage instructions. |
| `Changelog.md` | Version-by-version history. |

## How the Two Data Modes Work
Both make_sample_trace.py and extract_traces.py produce the same JSON schema — index.html doesn't know or care which one made the file it's showing. The synthetic generator hand-derives every stage so the math is internally consistent (scaling a dot_product value by √64 and softmaxing it really does produce the matching softmax value in the same file), even though the underlying weights are fabricated. extract_traces.py produces the identical shape, but every number comes from a real forward pass — including reconstructing the pre-softmax attention scores by replaying GPT-2's own attention math on a hook into c_attn, since HuggingFace only exposes the post-softmax result by default.
- [x] Real trace extraction, shape tracker, click-to-trace, generation loop, guided walkthrough, and pipeline map.
- [ ] Embedding-space nearest-neighbor explorer (cosine similarity over GPT-2's 50,257 × 768 table).
- [ ] Save/share a specific run via URL.
- [ ] "Classic vs. modern architecture" side-by-side: GPT-2 vs. Qwen2.5-0.5B (RoPE, RMSNorm, SwiGLU, GQA), as its own linked page.

See [Changelog.md](Changelog.md) for what's shipped so far.

## Acknowledgments
Directly inspired by Georgia Tech/IBM's Transformer Explainer (paper, MIT-licensed) — this project reimplements the same underlying concepts with original code, copy, and design, plus its own additions (worked arithmetic, click-to-trace, generation loop, guided walkthrough). The induction-head detection is based on Olsson et al., "In-context Learning and Induction Heads" (Anthropic, 2022).

## License

MIT — see [LICENSE](LICENSE).

## Author

Built by Yash Tekavade.
[GitHub](https://github.com/yashtekavade) · LinkedIn · Substack · Portfolio

