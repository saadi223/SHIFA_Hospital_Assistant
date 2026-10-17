"""
chunker.py  (Phase 4)
Document_loader se aaye "page records" ko chhote, overlapping chunks mein todta hai,
aur har chunk ke saath metadata + unique chunk_id jodta hai.
"""

import re

CHUNK_SIZE = 350     # characters per chunk (kept small: the embedding model reads only ~128 tokens)
CHUNK_OVERLAP = 50   # characters repeated between consecutive chunks


# ---------------------------------------------------------------
# 1) Text ko paragraphs aur sentences mein todne ke helpers
# ---------------------------------------------------------------
def split_into_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def split_into_sentences(text):
    # Simple sentence splitter: '.', '!', '?' ke baad space + capital letter/number par todo.
    # Hospital policy text ke liye ye kaafi achha kaam karta hai.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------
# 2) Paragraphs ki list ko CHUNK_SIZE ke hisaab se jodo/todo
# ---------------------------------------------------------------
def pack_paragraphs(paragraphs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
                     split_long_by_sentence=True):
    chunks = []
    current = ""

    for para in paragraphs:
        # Agar ek paragraph akela hi chunk_size se bada hai...
        if len(para) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            if split_long_by_sentence:
                # ...to use sentences mein tod kar normal tarike se pack karo
                sentences = split_into_sentences(para)
                for sent in sentences:
                    if current and len(current) + len(sent) + 1 > chunk_size:
                        chunks.append(current.strip())
                        current = current[-overlap:] + " " + sent
                    else:
                        current = (current + " " + sent).strip()
            else:
                # CSV rows: pura row ek chunk mein hi rehne do (todo mat)
                chunks.append(para.strip())
            continue

        # Normal paragraph: current chunk mein fit hota hai to jodo, warna naya chunk shuru karo
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current.strip())
            current = current[-overlap:] + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


# ---------------------------------------------------------------
# 3) Ek page-record ko chunk-records mein badlo
# ---------------------------------------------------------------
def chunk_record(record, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = split_into_paragraphs(record["text"])

    # CSV ke liye: har row apne aap mein poora rehna chahiye, isliye sentence-split band
    split_long = record["file_type"] != "csv"

    pieces = pack_paragraphs(paragraphs, chunk_size, overlap, split_long_by_sentence=split_long)

    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append({
            "chunk_id": f"{record['doc_id']}_p{record['page']}_c{i}",
            "filename": record["filename"],
            "doc_id": record["doc_id"],
            "title": record["title"],
            "department": record["department"],
            "file_type": record["file_type"],
            "page": record["page"],
            "chunk_index_in_page": i,
            "text": piece,
        })
    return chunks


# ---------------------------------------------------------------
# 4) Saare records ek saath chunk karo
# ---------------------------------------------------------------
def chunk_all_records(records, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    all_chunks = []
    for record in records:
        all_chunks.extend(chunk_record(record, chunk_size, overlap))
    return all_chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from document_loader import load_all_documents

    recs = load_all_documents("documents")
    chks = chunk_all_records(recs)
    print(f"{len(recs)} page-records -> {len(chks)} chunks")
