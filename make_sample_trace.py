"""
make_sample_trace.py (v3)

Adds worked-example arithmetic on top of v2's attention pipeline:
  - value_sample per head: dim-0 of that head's Value vector for every
    token position, so Attention x Value = Out can be computed for real
    (dynamically, in the browser) for whichever head/token is selected.
  - expansion_example per layer: boundary terms + result for the MLP's
    Embeddings x Weights + Bias = Expanded formula.
  - logits_example: boundary terms + result for the final
    Embedding x Projection Weights + Bias = Logits formula.

As before: shapes and relationships are real, the underlying numbers are
synthetic. Swap in extract_traces.py's output for the real thing.
"""

import json
import math
import random

random.seed(7)

N_LAYER = 12
N_HEAD = 12
D_MODEL = 768
D_HEAD = D_MODEL // N_HEAD  # 64
D_MLP = 3072
VOCAB = 50257

# New example prompt (mirrors the repeat-structure of the previous "cat" example
# so the induction head demo still works, but with different content).
TOKENS = ["The", " rocket", " climbed", " through", " the", " clouds", ".",
          " The", " rocket", " climbed", " through"]
N = len(TOKENS)
INDUCTION_LAYER, INDUCTION_HEAD = 9, 5

TOKEN_IDS = [abs(hash(t)) % 50257 for t in TOKENS]
POSITIONS = list(range(N))


def causal_softmax_matrix(n, peak_first_token=0.15, seed_offset=0):
    rnd = random.Random(seed_offset)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        raw = [rnd.random() for _ in range(i + 1)]
        raw[0] += peak_first_token * (i + 1)
        total = sum(raw)
        for j in range(i + 1):
            mat[i][j] = raw[j] / total
    return mat


def induction_softmax_matrix(n):
    mat = [[0.0] * n for _ in range(n)]
    seen = {}
    for i in range(n):
        if i == 0:
            mat[i][0] = 1.0
            seen[TOKENS[i]] = i
            continue
        tok = TOKENS[i]
        if tok in seen and seen[tok] < i:
            target = seen[tok] + 1
            for j in range(i + 1):
                mat[i][j] = 0.02
            mat[i][target] = 1.0 - 0.02 * i
        else:
            for j in range(i + 1):
                mat[i][j] = 0.05
            mat[i][i] = 1.0 - 0.05 * i
        seen[tok] = i
        row_sum = sum(mat[i][: i + 1])
        mat[i] = [v / row_sum if idx <= i else 0.0 for idx, v in enumerate(mat[i])]
    return mat


def derive_pipeline(softmax_mat, seed_offset):
    n = len(softmax_mat)
    rnd = random.Random(seed_offset + 999)
    scaled = [[None] * n for _ in range(n)]
    dot = [[None] * n for _ in range(n)]
    for i in range(n):
        row = softmax_mat[i][: i + 1]
        max_w = max(row)
        for j in range(i + 1):
            w = max(softmax_mat[i][j], 1e-9)
            s = math.log(w / max_w) + 3.0
            s = max(s, -6.0)
            scaled[i][j] = round(s, 2)
            dot[i][j] = round(s * math.sqrt(D_HEAD), 2)
        for j in range(i + 1, n):
            dot[i][j] = round(rnd.uniform(-15, 15), 2)
            scaled[i][j] = None
    return dot, scaled


def value_sample(n, seed_offset):
    """dim-0 of the Value vector for each token position, for this head."""
    rnd = random.Random(seed_offset + 5000)
    return [round(rnd.uniform(-1.4, 1.4), 3) for _ in range(n)]


layers = []
for L in range(N_LAYER):
    heads = []
    for H in range(N_HEAD):
        if L == INDUCTION_LAYER and H == INDUCTION_HEAD:
            softmax_mat = induction_softmax_matrix(N)
            label = "induction-like"
        else:
            softmax_mat = causal_softmax_matrix(N, seed_offset=L * 100 + H)
            label = None
        dot, scaled = derive_pipeline(softmax_mat, seed_offset=L * 100 + H)
        heads.append({
            "head": H,
            "dot_product": dot,
            "scaled_masked": scaled,
            "softmax": softmax_mat,
            "value_sample": value_sample(N, seed_offset=L * 100 + H),
            "label": label,
        })

    act_mean = round(0.4 + 0.05 * (L % 4), 3)
    act_max = round(3.0 + 0.3 * (L % 5), 3)
    rnd = random.Random(L + 42)
    emb_first, emb_last = round(rnd.uniform(-0.6, 0.6), 3), round(rnd.uniform(-0.6, 0.6), 3)
    w_first, w_last = round(rnd.uniform(-0.05, 0.05), 4), round(rnd.uniform(-0.05, 0.05), 4)
    bias = round(rnd.uniform(-0.1, 0.1), 3)
    result = round(rnd.uniform(0, act_max), 3)  # plausible, within this layer's activation range
    mlp = {
        "expand_dim": D_MLP,
        "activation_mean": act_mean,
        "activation_max": act_max,
        "pct_active_gelu": round(0.55 + 0.02 * (L % 6), 3),
        "expansion_example": {
            "token_index": N - 1,
            "emb_first": emb_first, "emb_last": emb_last,
            "w_first": w_first, "w_last": w_last,
            "bias": bias, "result": result,
        },
    }
    layers.append({"layer": L, "attention": {"heads": heads}, "mlp": mlp})

candidates = [
    (" the", 0.31), (" a", 0.10), (" his", 0.07), (" top", 0.05), (" it", 0.05),
    (" my", 0.04), (" that", 0.04), (" this", 0.03), (" all", 0.03), (" its", 0.03),
    (" one", 0.03), (" some", 0.02), (" clouds", 0.02), (" sky", 0.02), (" air", 0.02),
]
remaining = 1.0 - sum(p for _, p in candidates)
candidates.append((" [other 50,242 tokens]", round(remaining, 3)))
logits = [(tok, round(math.log(max(p, 1e-6)) + 10, 4)) for tok, p in candidates]

rnd = random.Random(999)
logits_example = {
    "emb_first": round(rnd.uniform(-0.5, 0.5), 3), "emb_last": round(rnd.uniform(-0.5, 0.5), 3),
    "w_first": round(rnd.uniform(-0.03, 0.03), 4), "w_last": round(rnd.uniform(-0.03, 0.03), 4),
    "bias": round(rnd.uniform(-0.2, 0.2), 3),
    "result": logits[0][1],  # tie to the real top logit so the story is consistent
}

trace = {
    "meta": {
        "model": "gpt2 (small) — SYNTHETIC DEMO DATA, not a real forward pass",
        "n_layer": N_LAYER, "n_head": N_HEAD, "d_model": D_MODEL, "d_head": D_HEAD,
        "vocab_size": VOCAB,
        "note": "Replace with real output from extract_traces.py",
    },
    "prompt": "".join(TOKENS),
    "tokens": TOKENS,
    "token_ids": TOKEN_IDS,
    "positions": POSITIONS,
    "layers": layers,
    "induction_heads": [{"layer": INDUCTION_LAYER, "head": INDUCTION_HEAD, "score": 0.94}],
    "output_logits": logits,
    "logits_example": logits_example,
}

with open("sample_trace.json", "w") as f:
    json.dump(trace, f)

print(f"Wrote sample_trace.json — {N} tokens, {N_LAYER} layers x {N_HEAD} heads, with worked-example arithmetic")
