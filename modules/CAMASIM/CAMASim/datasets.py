"""
datasets.py — thin dataset loader for the unified evaluate() driver.

Turns a named retrieval dataset into document/query embeddings, so a user can
say ``dataset: {"name": "scifact"}`` and let the pipeline do the embedding.

This is intentionally a SEPARATE, thin layer over the embeddings-core in
evaluate.py: the heavy stack (sentence-transformers, BEIR, torch) is imported
LAZILY inside the function, so importing the core / running the fast unit tests
never drags in those dependencies.
"""

from __future__ import annotations

from typing import Optional, Tuple


def load_beir_embeddings(
    name: str = "scifact",
    data_path: Optional[str] = None,
    num_docs: Optional[int] = None,
    num_queries: Optional[int] = None,
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
    verbose: bool = True,
    cache_path: Optional[str] = None,
) -> Tuple["np.ndarray", "np.ndarray", dict]:
    """Load a BEIR dataset and embed it with a sentence-transformer.

    Parameters
    ----------
    name        : BEIR dataset name (e.g. "scifact", "nfcorpus", "fiqa").
    data_path   : folder holding the dataset (default: "datasets/<name>").
    num_docs    : cap on documents (None = all).
    num_queries : cap on queries (None = all).
    model_name  : sentence-transformers model (default all-MiniLM-L6-v2, 384-dim).
    normalize   : L2-normalise embeddings (so inner product = cosine).
    cache_path  : if given, embeddings are cached to this .npz. On a later call
                  with the SAME parameters the vectors are loaded straight from
                  disk -- WITHOUT importing torch/sentence-transformers -- so the
                  slow one-time embedding is never repeated.

    Returns
    -------
    (docs, queries, qrels) : doc embeddings (Nd, dim) float32,
                             query embeddings (Nq, dim) float32,
                             qrels dict (relevance labels, may be empty).
    """
    import json
    import os
    import numpy as np   # numpy is light; torch stays lazy on the miss path below

    def _say(msg):
        if verbose:
            print(f"[datasets] {msg}", flush=True)

    if cache_path and not cache_path.endswith(".npz"):
        cache_path += ".npz"

    meta = {"name": name, "num_docs": num_docs, "num_queries": num_queries,
            "model_name": model_name, "normalize": bool(normalize)}

    # --- cache HIT: load vectors with NO heavy imports (no torch at all) ---
    if cache_path and os.path.exists(cache_path):
        try:
            z = np.load(cache_path, allow_pickle=False)
            if json.loads(z["meta"].item()) == meta:
                docs, queries = z["docs"], z["queries"]
                qrels = {}
                side = cache_path + ".qrels.json"
                if os.path.exists(side):
                    with open(side) as f:
                        qrels = json.load(f)
                _say(f"cache hit: loaded {cache_path} (docs {docs.shape}, "
                     f"queries {queries.shape}) -- no embedding / no torch")
                return docs, queries, qrels
            _say(f"cache {cache_path} exists but parameters changed; re-embedding")
        except Exception as e:
            _say(f"could not read cache {cache_path} ({e}); re-embedding")

    # --- cache MISS: the heavy path (lazy torch/transformers import) ---
    _say("importing embedding stack (torch/transformers -- this can take a bit)...")
    import time as _time
    from sentence_transformers import SentenceTransformer
    from beir.datasets.data_loader import GenericDataLoader

    data_path = data_path or f"datasets/{name}"
    _say(f"loading BEIR dataset '{name}' from {data_path} ...")
    corpus, queries_raw, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    q_ids = list(queries_raw.keys())
    if num_queries is not None:
        q_ids = q_ids[:num_queries]
    q_texts = [queries_raw[q] for q in q_ids]

    d_ids = list(corpus.keys())
    if num_docs is not None:
        d_ids = d_ids[:num_docs]
    d_texts = [corpus[d].get("title", "") + " " + corpus[d].get("text", "") for d in d_ids]
    _say(f"{len(d_texts)} docs, {len(q_texts)} queries to embed")

    _say(f"loading model '{model_name}' (first run downloads ~80MB, then caches)...")
    t0 = _time.time()
    model = SentenceTransformer(model_name)
    _say(f"model ready in {_time.time()-t0:.1f}s")

    _say("embedding documents...")
    docs = model.encode(d_texts, batch_size=64, normalize_embeddings=normalize,
                        show_progress_bar=verbose).astype(np.float32)
    _say("embedding queries...")
    queries = model.encode(q_texts, batch_size=64, normalize_embeddings=normalize,
                          show_progress_bar=verbose).astype(np.float32)
    _say(f"done: docs {docs.shape}, queries {queries.shape}")

    # --- save cache so future runs skip all of the above ---
    if cache_path:
        d = os.path.dirname(os.path.abspath(cache_path))
        os.makedirs(d, exist_ok=True)
        np.savez(cache_path, docs=docs, queries=queries, meta=json.dumps(meta))
        try:
            with open(cache_path + ".qrels.json", "w") as f:
                json.dump(qrels, f)
        except TypeError:
            pass
        _say(f"cached embeddings to {cache_path} (future runs skip embedding + torch)")

    return docs, queries, qrels


def resolve_dataset(dataset_cfg) -> Tuple["np.ndarray", "np.ndarray", Optional[dict]]:
    """Resolve a config's ``dataset`` entry into (docs, queries, qrels).

    Accepts either a dataset name (str) or a dict of loader kwargs, e.g.
    {"name": "scifact", "num_docs": 5000, "num_queries": 200}.
    """
    if isinstance(dataset_cfg, str):
        return load_beir_embeddings(name=dataset_cfg)
    if isinstance(dataset_cfg, dict) and "name" in dataset_cfg:
        return load_beir_embeddings(**dataset_cfg)
    raise ValueError(
        "config['dataset'] must be a dataset name or a dict with a 'name' key "
        "(or pass docs=/queries= embeddings directly to evaluate())."
    )
