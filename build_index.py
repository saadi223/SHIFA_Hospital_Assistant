"""
build_index.py  (Phase 6)
Hospital documents se poora knowledge base banata hai aur SAVE karta hai:
  knowledge_base/faiss.index   -> FAISS search structure
  knowledge_base/chunks.pkl    -> har chunk ka text + metadata (FAISS position se match)
  knowledge_base/metadata.pkl  -> build info (kab bana, kitne chunks, konsa model)

Isko sirf ek baar chalana hai (ya jab documents badlein). Streamlit app (app.py) ye
saved files sirf LOAD karega, documents ko dobara process nahi karega.
"""

import pickle
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np

from document_loader import load_all_documents
from chunker import chunk_all_records
from embedder import embed_chunks, check_token_lengths, MODEL_NAME

DOCS_DIR = Path("documents")
KB_DIR = Path("knowledge_base")


def build_and_save_index(docs_dir=DOCS_DIR, kb_dir=KB_DIR):
    kb_dir = Path(kb_dir)
    kb_dir.mkdir(parents=True, exist_ok=True)

    # 1. Documents load + clean karo
    print("Step 1/4: Documents load ho rahi hain...")
    records = load_all_documents(docs_dir)
    print(f"   {len(records)} page-records mile")

    # 2. Chunks banao
    print("Step 2/4: Chunking ho rahi hai...")
    chunks = chunk_all_records(records)
    print(f"   {len(chunks)} chunks bane")

    # The embedding model silently cuts chunks longer than its token limit. Check it now.
    limit, longest, over = check_token_lengths([c["text"] for c in chunks])
    print(f"   Token check: limit={limit}, longest chunk={longest} tokens, chunks over limit={over}")
    if over:
        print("   WARNING: some chunks are too long and will be cut. Lower CHUNK_SIZE in chunker.py.")

    # 3. Embeddings banao
    print("Step 3/4: Embeddings ban rahi hain (ek baar ka kaam)...")
    embeddings = embed_chunks(chunks)
    embeddings = np.array(embeddings).astype("float32")   # FAISS ko float32 chahiye
    print(f"   Embeddings shape: {embeddings.shape}")

    # 4. FAISS index banao
    #    IndexFlatIP = Inner Product search. Kyunki embeddings normalized hain
    #    (Phase 5 mein normalize_embeddings=True), Inner Product == Cosine Similarity.
    #    "Flat" matlab exact search (approximate nahi) - itne kam chunks ke liye perfect hai.
    print("Step 4/4: FAISS index ban raha hai...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"   Index mein {index.ntotal} vectors hain")

    # chunk se "embedding" key hata do (wo ab FAISS mein hai, dobara save karne ki zaroorat nahi)
    chunks_to_save = []
    for c in chunks:
        c = dict(c)
        c.pop("embedding", None)
        chunks_to_save.append(c)

    # 5. Save: FAISS index
    faiss.write_index(index, str(kb_dir / "faiss.index"))

    # 6. Save: chunks (text + metadata, FAISS position ke order mein)
    with open(kb_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks_to_save, f)

    # 7. Save: metadata (debugging/reference ke liye)
    meta = {
        "built_at": datetime.now().isoformat(),
        "num_documents": len({c["doc_id"] for c in chunks_to_save}),
        "num_chunks": len(chunks_to_save),
        "embedding_model": MODEL_NAME,
        "embedding_dimension": dimension,
        "index_type": "IndexFlatIP",
    }
    with open(kb_dir / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("\n✅ Knowledge base ban gaya aur save ho gaya:")
    for key, value in meta.items():
        print(f"   {key}: {value}")

    return index, chunks_to_save, meta


if __name__ == "__main__":
    build_and_save_index()
