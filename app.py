"""
app.py  (Phase 10)
SHIFA Knowledge Assistant - Streamlit chat UI.

IMPORTANT: Ye app documents ko kabhi re-process/re-embed NAHI karta.
Ye sirf PRE-BUILT knowledge_base/ (faiss.index + chunks.pkl) ko LOAD karta hai,
jo build_index.py se Colab mein pehle hi bana chuka hai.
"""

import streamlit as st

from retriever import load_knowledge_base
from rag_pipeline import get_groq_client, stream_answer
from embedder import get_model as load_embedding_model


# ---------------------------------------------------------------
# Page config (sabse pehle call hona chahiye)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="SHIFA Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
)


# ---------------------------------------------------------------
# Cached loaders - ye sirf EK BAAR chalte hain, har interaction par nahi
# ---------------------------------------------------------------
@st.cache_resource
def get_knowledge_base():
    """Saved FAISS index + chunks load karta hai. Documents re-process NAHI hote."""
    return load_knowledge_base()


@st.cache_resource
def get_embedding_model():
    """Loads the embedding model once and keeps it in cache."""
    return load_embedding_model()


@st.cache_resource
def get_client():
    """Groq client banata hai, API key Streamlit secrets se leta hai."""
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY nahi mili. .streamlit/secrets.toml mein daalo (Colab/local) "
                  "ya Streamlit Cloud ke Secrets settings mein daalo.")
        st.stop()
    return get_groq_client(api_key)


# ---------------------------------------------------------------
# App start: knowledge base + model + client load karo
# ---------------------------------------------------------------
with st.spinner("Knowledge base load ho raha hai..."):
    try:
        index, chunks_db = get_knowledge_base()
    except (RuntimeError, FileNotFoundError) as e:   # e.g. index built with an older embedding model
        st.error(f"Knowledge base could not be loaded: {e}")
        st.stop()
    get_embedding_model()          # isse load karke cache mein daal do, retriever isko use karega
    client = get_client()


# ---------------------------------------------------------------
# Sidebar: hospital info aur disclaimer
# ---------------------------------------------------------------
with st.sidebar:
    st.title("🏥 SHIFA Knowledge Assistant")
    st.caption("Enterprise RAG Assistant for Hospital Staff")

    st.warning("⚠️ DEMO / SYNTHETIC HOSPITAL DATA — NOT FOR REAL CLINICAL USE", icon="⚠️")

    st.markdown("---")
    st.markdown(f"**Knowledge base:**")
    st.markdown(f"- {len(chunks_db)} chunks indexed")
    departments = sorted({c["department"] for c in chunks_db})
    st.markdown(f"- Departments: {', '.join(departments)}")

    st.markdown("---")
    st.markdown("**Example questions:**")
    example_questions = [
        "What is the procedure for admitting an emergency patient?",
        "What is the deposit for ICU admission?",
        "What should I do after a needle-stick injury?",
        "How long do insulin doses need independent double-check?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ---------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("Ask SHIFA Knowledge Assistant")
st.caption("Ask a question about hospital policies and procedures. Answers are grounded in official documents only.")


# ---------------------------------------------------------------
# Purana chat history dikhao
# ---------------------------------------------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"**[{s['tag']}] {s['filename']}** — {s['department']}, Page {s['page']} "
                                f"(relevance: {s['score']:.2f})")
                    st.caption(f"\"{s['excerpt']}\"")


# ---------------------------------------------------------------
# Naya sawaal lo (ya sidebar se example click hua ho)
# ---------------------------------------------------------------
question = st.chat_input("Type your question here...")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    # User ka message dikhao + save karo
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate the answer
    with st.chat_message("assistant"):
        # The spinner covers only retrieval; streaming starts after it closes
        with st.spinner("Documents search ho rahe hain..."):
            stream, state = stream_answer(question, index, chunks_db, client)

        st.write_stream(stream)            # text appears token by token
        answer_text = state["full_text"]   # the generator has finished, so this is complete
        cited_sources = state["cited_sources"]

        if state["error"]:
            st.error(f"The answer was interrupted ({state['error']}). Please try again.")
            answer_text += "\n\n_The answer was interrupted._"

        if cited_sources:
            with st.expander(f"📚 Sources ({len(cited_sources)})"):
                for s in cited_sources:
                    st.markdown(f"**[{s['tag']}] {s['filename']}** — {s['department']}, Page {s['page']} "
                                f"(relevance: {s['score']:.2f})")
                    st.caption(f"\"{s['excerpt']}\"")

    # Assistant ka message save karo
    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer_text,
        "sources": cited_sources,
    })
