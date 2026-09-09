"""
extract_traces.py (v3)

Runs a REAL GPT-2 (small) forward pass and dumps trace.json with:
  - the full per-head attention pipeline (dot_product -> scaled_masked -> softmax)
  - real Value vectors (dim 0 per token, per head) so Attention x Value = Out
    can be computed exactly in the browser
  - a real worked example for the MLP's Embeddings x Weights + Bias = Expanded
  - a real worked example for the final Embedding x Projection Weights = Logits

HuggingFace's output_attentions=True only gives you the post-softmax
result. To get the raw QK^T scores and the scaled+masked scores in
between, this script hooks `block.attn.c_attn` (which produces the
concatenated Q/K/V projection) and manually replays GPT-2's own attention
math on it: split heads -> Q @ K^T -> /sqrt(head_dim) -> causal mask ->
softmax. The final softmax result should match output_attentions almost
exactly (small floating point differences are expected) — the script
asserts this as a sanity check.

Run locally (needs internet to pull the GPT-2 checkpoint from Hugging
Face — won't work in a sandboxed environment with restricted egress):

    pip install -r requirements.txt
    python extract_traces.py --prompt "The rocket climbed through the clouds. The rocket climbed through" --probe-induction
"""

import argparse
import json
import math
import random

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    return tok, model


def split_heads(x, n_head, head_dim):
    # x: (batch, seq, n_embd) -> (batch, n_head, seq, head_dim)
    new_shape = x.size()[:-1] + (n_head, head_dim)
    return x.view(*new_shape).permute(0, 2, 1, 3)


def replay_attention(qkv, n_head, head_dim, n_embd):
    """Manually reproduces GPT-2's attention math from the raw c_attn output,
    returning (dot_product, scaled_masked, softmax, v_heads)."""
    q, k, v = qkv.split(n_embd, dim=2)
    q = split_heads(q, n_head, head_dim)[0]  # (n_head, seq, head_dim)
    k = split_heads(k, n_head, head_dim)[0]
    v = split_heads(v, n_head, head_dim)[0]
    seq = q.shape[1]

    dot = torch.matmul(q, k.transpose(-1, -2))  # (n_head, seq, seq) — raw QK^T
    scaled = dot / math.sqrt(head_dim)

    causal_mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    mask_value = torch.finfo(scaled.dtype).min
    masked = scaled.masked_fill(causal_mask.unsqueeze(0), mask_value)
    softmax_w = torch.softmax(masked, dim=-1)
    return dot, scaled, causal_mask, softmax_w, v


def qkv_hook_factory(store, layer_idx):
    def hook(module, inp, out):
        store[layer_idx] = out.detach()
    return hook


def mlp_io_hook_factory(store, layer_idx):
    """Captures both the MLP's input (ln_2(x), what actually gets multiplied
    by c_fc's weights) and its pre-GELU output (768 -> 3072)."""
    def hook(module, inp, out):
        store[layer_idx] = {"input": inp[0].detach(), "output": out.detach()}
    return hook


def extract_trace(prompt: str, tok, model):
    inputs = tok(prompt, return_tensors="pt")
    token_ids = inputs["input_ids"][0]
    tokens = [tok.decode([t]) for t in token_ids]
    n = len(tokens)

    qkv_store, mlp_store = {}, {}
    handles = []
    for i, block in enumerate(model.transformer.h):
        handles.append(block.attn.c_attn.register_forward_hook(qkv_hook_factory(qkv_store, i)))
        handles.append(block.mlp.c_fc.register_forward_hook(mlp_io_hook_factory(mlp_store, i)))

    with torch.no_grad():
        out = model(**inputs, output_attentions=True, output_hidden_states=True)

    for h in handles:
        h.remove()

    n_layer, n_head = model.config.n_layer, model.config.n_head
    head_dim = model.config.n_embd // n_head
    n_embd = model.config.n_embd

    layers = []
    for L in range(n_layer):
        dot, scaled, causal_mask, softmax_w, v_heads = replay_attention(qkv_store[L], n_head, head_dim, n_embd)

        hf_attn = out.attentions[L][0]
        max_diff = (softmax_w - hf_attn).abs().max().item()
        if max_diff > 1e-3:
            print(f"  [warn] layer {L}: replayed softmax differs from HF by {max_diff:.4f} (expected ~0)")

        heads = []
        for H in range(n_head):
            dot_h = [[round(v, 2) for v in row] for row in dot[H].tolist()]
            scaled_h = [
                [None if causal_mask[i, j] else round(scaled[H, i, j].item(), 2) for j in range(n)]
                for i in range(n)
            ]
            softmax_h = [[round(v, 4) for v in row] for row in softmax_w[H].tolist()]
            value_sample = [round(v_heads[H, k, 0].item(), 3) for k in range(n)]
            heads.append({
                "head": H, "dot_product": dot_h, "scaled_masked": scaled_h,
                "softmax": softmax_h, "value_sample": value_sample, "label": None,
            })

        # ---- MLP expansion worked example (exact, using the real last-token vector) ----
        mlp_in = mlp_store[L]["input"][0]   # (seq, n_embd) — what actually feeds c_fc
        mlp_out = mlp_store[L]["output"][0]  # (seq, d_mlp) — pre-GELU
        c_fc_weight = model.transformer.h[L].mlp.c_fc.weight  # Conv1D: (n_embd, d_mlp)
        c_fc_bias = model.transformer.h[L].mlp.c_fc.bias
        t_idx = n - 1
        expansion_example = {
            "token_index": t_idx,
            "emb_first": round(mlp_in[t_idx, 0].item(), 3),
            "emb_last": round(mlp_in[t_idx, -1].item(), 3),
            "w_first": round(c_fc_weight[0, 0].item(), 4),
            "w_last": round(c_fc_weight[-1, 0].item(), 4),
            "bias": round(c_fc_bias[0].item(), 3),
            "result": round(mlp_out[t_idx, 0].item(), 3),  # exact — computed over all 768 terms
        }

        act = mlp_out
        layers.append({
            "layer": L,
            "attention": {"heads": heads},
            "mlp": {
                "expand_dim": act.shape[-1],
                "activation_mean": round(act.mean().item(), 3),
                "activation_max": round(act.max().item(), 3),
                "pct_active_gelu": round((act > 0).float().mean().item(), 3),
                "expansion_example": expansion_example,
            },
        })

    logits = out.logits[0, -1]
    top = torch.topk(logits, 15)
    output_logits = [[tok.decode([idx.item()]), round(val.item(), 4)] for val, idx in zip(top.values, top.indices)]

    # ---- Final logits worked example (exact) ----
    final_hidden = out.hidden_states[-1][0, -1]  # (n_embd,) — post-ln_f, what lm_head consumes
    lm_weight = model.lm_head.weight  # (vocab, n_embd), tied to token embeddings
    top_id = top.indices[0].item()
    logits_example = {
        "emb_first": round(final_hidden[0].item(), 3),
        "emb_last": round(final_hidden[-1].item(), 3),
        "w_first": round(lm_weight[top_id, 0].item(), 4),
        "w_last": round(lm_weight[top_id, -1].item(), 4),
        "bias": 0.0,  # GPT-2's lm_head is bias-free (weight-tied to the embedding table)
        "result": round(top.values[0].item(), 4),  # exact
    }

    return {
        "meta": {
            "model": "gpt2 (small) — REAL forward pass",
            "n_layer": n_layer, "n_head": n_head, "d_model": n_embd,
            "d_head": head_dim, "vocab_size": model.config.vocab_size,
        },
        "prompt": prompt,
        "tokens": tokens,
        "token_ids": token_ids.tolist(),
        "positions": list(range(n)),
        "layers": layers,
        "induction_heads": [],
        "output_logits": output_logits,
        "logits_example": logits_example,
    }


def generate_chain(prompt: str, tok, model, n_steps=6, top_n=5):
    """Real autoregressive generation, greedy, n_steps tokens — matching the
    generation_chain schema the frontend's 'Watch it generate' panel expects.
    Each step records the top-n candidates *before* the chosen token was
    appended, so the panel can show what the model was actually weighing."""
    input_ids = tok(prompt, return_tensors="pt")["input_ids"]
    chain = []
    for _ in range(n_steps):
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, top_n)
        candidates = [[tok.decode([idx.item()]), round(val.item(), 4)] for val, idx in zip(top.values, top.indices)]
        next_id = top.indices[0].item()  # greedy
        chain.append({"appended": tok.decode([next_id]), "candidates": candidates})
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
    return chain


def probe_induction_heads(tok, model, vocab_sample=2000, seq_len=25, n_trials=8, top_n=10):
    n_layer, n_head = model.config.n_layer, model.config.n_head
    scores = torch.zeros(n_layer, n_head)
    for _ in range(n_trials):
        seq = [random.randint(0, vocab_sample) for _ in range(seq_len)]
        input_ids = torch.tensor([seq + seq])
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)
        for L in range(n_layer):
            attn = out.attentions[L][0]
            for H in range(n_head):
                total, count = 0.0, 0
                for i in range(seq_len, 2 * seq_len):
                    target = i - seq_len + 1
                    if target < i:
                        total += attn[H, i, target].item()
                        count += 1
                if count:
                    scores[L, H] += total / count
    scores /= n_trials
    flat = [(scores[L, H].item(), L, H) for L in range(n_layer) for H in range(n_head)]
    flat.sort(reverse=True)
    return [{"layer": L, "head": H, "score": round(s, 4)} for s, L, H in flat[:top_n]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The rocket climbed through the clouds. The rocket climbed through")
    parser.add_argument("--probe-induction", action="store_true")
    parser.add_argument("--generate-steps", type=int, default=6, help="how many tokens to generate for the 'Watch it generate' panel")
    parser.add_argument("--out", default="trace.json")
    args = parser.parse_args()

    tok, model = load_model()
    trace = extract_trace(args.prompt, tok, model)

    if args.probe_induction:
        print("Probing all heads for induction behavior (takes a minute)...")
        ranked = probe_induction_heads(tok, model)
        trace["induction_heads"] = [h for h in ranked if h["score"] > 0.15]
        print("Top induction-like heads:")
        for h in ranked[:10]:
            print(f"  layer {h['layer']:>2}  head {h['head']:>2}   score={h['score']}")

    print(f"Generating {args.generate_steps} real tokens for the generation-loop panel...")
    trace["generation_chain"] = generate_chain(args.prompt, tok, model, n_steps=args.generate_steps)

    with open(args.out, "w") as f:
        json.dump(trace, f)
    print(f"Wrote {args.out} — {len(trace['tokens'])} tokens, {trace['meta']['n_layer']} layers")
