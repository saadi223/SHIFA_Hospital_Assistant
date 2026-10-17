"""
rag_pipeline.py  (Phase 8 + Phase 9)
Retrieved chunks ko context mein badalta hai, strict grounding instruction ke saath
Groq LLM ko bhejta hai, jawaab wapas deta hai, aur sirf un sources ki structured
citation list banata hai jo LLM ne asal mein apne jawaab mein use kiye.
"""

import os
import re
from groq import Groq

from retriever import load_knowledge_base, retrieve

GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_TOP_K = 4
TEMPERATURE = 0.1
# Hinglish uses more tokens than English, so this is higher than before (was 600).
# Trade-off: longer possible answers = more latency/cost. If the answer gets cut
# off, raise it. (Reasoning models may also count thinking tokens in this limit.)
MAX_TOKENS = 800

SYSTEM_PROMPT = """You are SHIFA Knowledge Assistant, a policy and procedure assistant for hospital staff at SHIFA General Hospital.

STRICT RULES:
1. Answer only from the provided hospital document context below. Do not use any outside knowledge.
2. If sufficient information is not available in the context, clearly say that the answer could not be found in the available documents. Do not guess or make up an answer.
3. When you use information from a source, refer to it using its tag, like [S1] or [S2]. Cite every factual claim.
4. This is a knowledge/policy assistant, not a medical diagnosis system. Do not give clinical advice beyond what is written in the policies.
5. Be concise and direct. Use short paragraphs or bullet points where helpful.
6. Reply in the same language and style as the staff question. English question -> English answer. Hinglish question (Hindi/Urdu written in Latin script) -> Hinglish answer in Latin script. Never use Devanagari or Urdu script.
7. Keep source tags like [S1], and all numbers, names, extension numbers, medicine names and units, exactly as they appear in the documents.
8. If you cannot find the answer (rule 2), say so in the language of the question.
"""

# Common Hinglish function words. English text almost never contains these.
HINGLISH_WORDS = {
    "kya", "hai", "hain", "hota", "hoti", "hote", "mein", "mai", "ka", "ki", "ke",
    "ko", "se", "par", "liye", "chahiye", "karna", "karte", "kare", "karein",
    "nahi", "nahin", "aur", "ya", "agar", "toh", "tak", "wala", "wali", "sakta",
    "sakte", "kaise", "kaisa", "kitna", "kitne", "kitni", "kab", "kaun", "kyun",
    "kyon", "padta", "padti", "lagta", "lagti", "batao", "bataiye", "jaye", "jaaye",
}

NO_ANSWER_MESSAGES = {
    "english": "The answer could not be found in the available documents.",
    "hinglish": "Iska jawaab available documents mein nahi mila.",
}


def detect_language(question):
    """Very simple check: 'hinglish' if the question has a common Hinglish word, else 'english'."""
    words = re.findall(r"[a-z]+", question.lower())
    return "hinglish" if any(w in HINGLISH_WORDS for w in words) else "english"


def no_answer_message(question):
    return NO_ANSWER_MESSAGES[detect_language(question)]


def get_groq_client(api_key=None):
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY nahi mili. Isse environment variable ya argument se pass karo.")
    return Groq(api_key=api_key)


def build_context(chunks):
    """
    Retrieved chunks ko numbered source blocks mein badalta hai, jaise:
    [S1] Source: EM-01_Emergency_Admission_Policy.pdf (Emergency, page 1)
    <chunk text>

    Har chunk ko uska tag ("S1", "S2"...) bhi assign karke wapas deta hai, taaki
    baad mein extract_cited_sources() isi mapping se match kar sake.
    """
    blocks = []
    tagged_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        tag = f"S{i}"
        header = f"[{tag}] Source: {chunk['filename']} ({chunk['department']}, page {chunk['page']})"
        blocks.append(f"{header}\n{chunk['text']}")
        chunk_with_tag = dict(chunk)
        chunk_with_tag["tag"] = tag
        tagged_chunks.append(chunk_with_tag)
    return "\n\n---\n\n".join(blocks), tagged_chunks


def extract_cited_sources(answer_text, tagged_chunks, excerpt_len=200):
    """
    Answer text mein se [S1], [S2] jaise tags dhoondhta hai, aur sirf unhi
    chunks ki structured citation list banata hai jo asal mein use hue.
    Agar LLM ne koi tag use hi nahi kiya (ya "not found" jawaab diya), khaali list aati hai.
    """
    used_tags = set(re.findall(r"\[S(\d+)\]", answer_text))   # jaise {"1", "3"}

    cited = []
    for chunk in tagged_chunks:
        tag_number = chunk["tag"].lstrip("S")
        if tag_number in used_tags:
            cited.append({
                "tag": chunk["tag"],
                "filename": chunk["filename"],
                "title": chunk["title"],
                "department": chunk["department"],
                "page": chunk["page"],
                "score": round(chunk["score"], 3),
                "excerpt": chunk["text"][:excerpt_len].strip() + ("..." if len(chunk["text"]) > excerpt_len else ""),
            })

    # Tag number ke hisaab se sort karo (S1, S2, S3...) taaki order saaf rahe
    cited.sort(key=lambda c: int(c["tag"].lstrip("S")))
    return cited


def build_messages(question, retrieved):
    """
    Shared by answer_question() and stream_answer() so the prompt can never drift apart.
    Returns (messages_for_groq, tagged_chunks).
    """
    context_text, tagged_chunks = build_context(retrieved)

    user_message = f"""Context from hospital documents:

{context_text}

---

Staff question: {question}

Answer the question using only the context above. Cite sources using their tags like [S1], [S2]."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return messages, tagged_chunks


def answer_question(question, index, chunks_db, client, top_k=DEFAULT_TOP_K, temperature=TEMPERATURE):
    """
    Full RAG flow (non-streaming): question -> retrieve -> Groq -> grounded answer -> cited sources.

    Returns:
    {
        "answer": str,                 # LLM answer, with [S1] tags
        "cited_sources": list,         # only the sources used in the answer
        "all_retrieved": list,         # all retrieved chunks (for debugging)
    }
    """
    # 1. Retrieve
    retrieved = retrieve(question, index, chunks_db, top_k=top_k)

    if not retrieved:
        return {
            "answer": no_answer_message(question),
            "cited_sources": [],
            "all_retrieved": [],
        }

    # 2. Build prompt (same helper as the streaming path)
    messages, tagged_chunks = build_messages(question, retrieved)

    # 3. Ask Groq
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )
    answer_text = response.choices[0].message.content

    # 4. Find which sources were cited
    cited_sources = extract_cited_sources(answer_text, tagged_chunks)

    return {
        "answer": answer_text,
        "cited_sources": cited_sources,
        "all_retrieved": tagged_chunks,
    }


def stream_answer(question, index, chunks_db, client, top_k=DEFAULT_TOP_K, temperature=TEMPERATURE):
    """
    Streaming version. Retrieval runs right away (so the caller can show a spinner
    only for this part). Returns (generator, state):

    - generator yields text pieces; pass it to st.write_stream().
    - state is filled while/after streaming:
        state["full_text"]      the text received so far (full answer when done)
        state["cited_sources"]  filled AFTER the stream ends, from the full text
        state["all_retrieved"]  tagged chunks
        state["error"]          None, or an error message if Groq failed
    A Groq error does not raise: the stream just stops and state["error"] is set,
    so the partial text stays visible.
    """
    state = {"full_text": "", "cited_sources": [], "all_retrieved": [], "error": None}

    retrieved = retrieve(question, index, chunks_db, top_k=top_k)

    if not retrieved:
        message = no_answer_message(question)

        def refusal_generator():
            state["full_text"] = message
            yield message

        return refusal_generator(), state

    messages, tagged_chunks = build_messages(question, retrieved)
    state["all_retrieved"] = tagged_chunks

    def token_generator():
        parts = []
        try:
            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                stream=True,
            )
            for event in stream:
                if not event.choices:
                    continue
                piece = event.choices[0].delta.content
                if piece:                      # skip empty / None deltas
                    parts.append(piece)
                    state["full_text"] = "".join(parts)
                    yield piece
        except Exception as e:                 # keep partial text, report the error
            state["error"] = f"{type(e).__name__}: {e}"
        finally:
            # Runs after the stream ends (or fails): parse source tags from the full text
            full_text = "".join(parts)
            state["full_text"] = full_text
            state["cited_sources"] = extract_cited_sources(full_text, tagged_chunks)

    return token_generator(), state


def format_sources_display(cited_sources):
    """Citation list ko ek readable text block mein format karta hai (Streamlit se pehle quick-check ke liye)."""
    if not cited_sources:
        return "(No sources cited)"

    lines = []
    for s in cited_sources:
        lines.append(f"[{s['tag']}] {s['filename']} | {s['department']} | Page {s['page']} | score={s['score']}")
        lines.append(f"    \"{s['excerpt']}\"")
    return "\n".join(lines)


if __name__ == "__main__":
    index, chunks_db = load_knowledge_base()
    client = get_groq_client()
    result = answer_question("What is the deposit for ICU admission?", index, chunks_db, client)
    print(result["answer"])
    print("\nSources:")
    print(format_sources_display(result["cited_sources"]))
