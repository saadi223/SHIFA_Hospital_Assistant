"""
embedder.py  (Phase 5)
Turns chunk text into embeddings using a sentence-transformers model.
The same model is used by build_index.py and by the app (retriever), so the
numbers stay comparable.

Model: paraphrase-multilingual-MiniLM-L12-v2 (multilingual, so English and
Hinglish questions can match English documents).
"""

from sentence_transformers import SentenceTransformer

# Single place for the model name. build_index.py saves it into metadata.pkl and
# retriever.py compares it with the saved value when loading the index.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model = None   # the model is loaded once and cached


def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME} (first time can take a while)...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts, batch_size=32, show_progress=True):
    """
    Takes a list of texts, returns a numpy array of embeddings.
    Example: embed_texts(["hello", "world"]) -> shape (2, 384)
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # needed so inner product == cosine similarity
    )
    return embeddings


def embed_chunks(chunks, batch_size=32):
    """Takes a list of chunk records and embeds the 'text' of each one."""
    texts = [c["text"] for c in chunks]
    return embed_texts(texts, batch_size=batch_size)


def check_token_lengths(texts):
    """
    The model silently cuts text longer than its max_seq_length (in tokens).
    Returns (limit, longest, number_over_limit) so build_index.py can warn.
    """
    model = get_model()
    limit = model.max_seq_length
    lengths = [len(model.tokenizer.encode(t)) for t in texts]
    over = sum(1 for n in lengths if n > limit)
    return limit, (max(lengths) if lengths else 0), over


if __name__ == "__main__":
    sample = ["What is the triage process?", "ICU room charges per day"]
    vectors = embed_texts(sample)
    print("Shape:", vectors.shape)
    print("max_seq_length:", get_model().max_seq_length)
