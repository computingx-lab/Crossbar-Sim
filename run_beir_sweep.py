import sys
sys.path.insert(0, "/mnt/c/Users/Charlie Power/CAMASim-Hybrid/modules/CAMASIM")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader

from CAMASim.function.module.distance import crossbar_innerproduct_pairwise
from CAMASim.function.module.sensing  import get_array_topk_results
from CAMASim.metrics.retrieval        import ideal_topk, recall_at_k, ndcg_at_k

# ── Load scifact ──────────────────────────────────────────────────────────────
data_path = "datasets/scifact"
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Use first 200 queries for speed
query_ids  = list(queries.keys())[:200]
query_texts = [queries[qid] for qid in query_ids]

doc_ids   = list(corpus.keys())
doc_texts = [corpus[did]['title'] + " " + corpus[did]['text'] for did in doc_ids]

print(f"Loaded {len(doc_texts)} docs, {len(query_texts)} queries")

# ── Embed with a small sentence transformer ───────────────────────────────────
model = SentenceTransformer('all-MiniLM-L6-v2')   # small, fast, 384-dim
print("Encoding docs...")
docs    = model.encode(doc_texts,    batch_size=64, show_progress_bar=True, normalize_embeddings=True)
print("Encoding queries...")
queries_emb = model.encode(query_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

docs        = docs.astype(np.float32)
queries_emb = queries_emb.astype(np.float32)

K = 10
gt = ideal_topk(docs, queries_emb, k=K)

# ── Sweep ─────────────────────────────────────────────────────────────────────
SIGMAS   = [0.0, 0.02, 0.05, 0.1, 0.2]
ADC_BITS = [None, 8, 6, 4]
LABELS   = {None: "Ideal ADC", 8: "8-bit", 6: "6-bit", 4: "4-bit"}

results = {}
for adc in ADC_BITS:
    recalls = []
    for sigma in SIGMAS:
        hw = {"sigma_conductance": sigma, "adc_bits": adc, "sigma_read": 0.0}
        scores  = crossbar_innerproduct_pairwise(docs, queries_emb, hw)
        pred, _ = get_array_topk_results(scores, k=K)
        r = recall_at_k(pred, gt)
        recalls.append(r)
        print(f"  {LABELS[adc]:<12}  sigma={sigma:.2f}  Recall@{K}={r:.3f}")
    results[adc] = recalls
    print()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for adc in ADC_BITS:
    ax.plot(SIGMAS, results[adc], marker='o', label=LABELS[adc])
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel("Conductance noise σ")
ax.set_ylabel(f"Recall@{K}")
ax.set_title(f"BEIR scifact: Recall@{K} vs Hardware Noise")
ax.set_xticks(SIGMAS)
ax.set_ylim(0.4, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("beir_recall_vs_noise.png", dpi=200, bbox_inches='tight')
print("Saved beir_recall_vs_noise.png")