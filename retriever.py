"""
retriever.py  (Phase 7)
Loads the saved FAISS index and chunks, and finds the Top-K most similar chunks
for a question. Used by the RAG pipeline and the Streamlit app.
"""

import pickle
from pathlib import Path

import faiss

from embedder import embed_texts, MODEL_NAME

KB_DIR = Path("knowledge_base")

# Minimum cosine similarity for a chunk to be kept.
# 0.0 means "only drop negative scores" (same as the old behaviour, where no
# threshold was passed). Do NOT guess a better value: run eval_retrieval.py,
# look at the scores, then set it here.
MIN_SCORE = 0.0

# We fetch this many times more candidates than top_k, filter, then cut to top_k.
CANDIDATE_MULTIPLIER = 3


def load_knowledge_base(kb_dir=KB_DIR):
    """
    Loads the saved FAISS index and chunks (text + metadata).
    Raises a clear error if the index was built with a different embedding model.
    Only load .pkl files that you created yourself (pickle can run code).
    """
    kb_dir = Path(kb_dir)

    meta_path = kb_dir / "metadata.pkl"
    if not meta_path.exists():
        raise RuntimeError(f"{meta_path} not found. Run: python build_index.py")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Both old and new models give 384-dim vectors, so FAISS would load an old
    # index without complaint and return wrong results. This check stops that.
    saved_model = meta.get("embedding_model")
    if saved_model != MODEL_NAME:
        raise RuntimeError(
            f"Index was built with embedding model '{saved_model}', but the code "
            f"now uses '{MODEL_NAME}'. Rebuild the index: python build_index.py"
        )

    index = faiss.read_index(str(kb_dir / "faiss.index"))
    with open(kb_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def retrieve(query, index, chunks, top_k=3, min_score=MIN_SCORE):
    """
    Finds the Top-K most similar chunks for one question.
    Each result has the chunk text, metadata and a similarity 'score'.
    """
    query_vector = embed_texts([query], show_progress=False).astype("float32")

    # Fetch extra candidates (but never more than the index holds)
    fetch_k = min(top_k * CANDIDATE_MULTIPLIER, index.ntotal)
    scores, indices = index.search(query_vector, fetch_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:                 # FAISS returns -1 when it has fewer results
            continue
        if score < min_score:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)

    # Results are already sorted by score, so just keep the best top_k
    return results[:top_k]


if __name__ == "__main__":
    index, chunks = load_knowledge_base()
    results = retrieve("What is the deposit for ICU admission?", index, chunks, top_k=3)
    for r in results:
        print(f"{r['score']:.3f} | {r['filename']} (p{r['page']}) | {r['text'][:100]}")
