"""
phase6_rag_app.py  —  Mushroom Expert RAG Chat Assistant
=========================================================
Retrieval-Augmented Generation chat using:
  - ChromaDB + sentence-transformers for knowledge retrieval
  - Google Gemini (AI Studio) for answer generation

Run:
    streamlit run phase6_rag_app.py

Setup required:
  1. python phase6_scrape.py
  2. python phase6_index.py
  3. Set GEMINI_API_KEY in .env file
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Mushroom Expert Chat",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths & settings ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_DIR     = os.path.join(BASE_DIR, "rag_db")
COLLECTION = "mushroom_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 5     # chunks retrieved per query
MAX_HISTORY = 6     # messages kept in prompt context (3 turns)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable mushroom expert assistant.
You specialize in:
- Oyster mushroom cultivation and substrate bag disease management
- Mushroom disease identification (Trichoderma, Aspergillus, Rhizopus, bacterial blotch, etc.)
- Mushroom species — edible, medicinal, poisonous
- Disease prevention, treatment, and cultivation best practices
- General mycology and mushroom biology

Use the provided context from authoritative sources. Be concise and practical.
Use bullet points and clear structure in your answers.
If the context is insufficient, use your general knowledge but note that.
Never make up specific numbers, research findings, or species classifications."""

# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_retriever():
    """Load ChromaDB collection and sentence-transformer embedder."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    client     = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION)
    embedder   = SentenceTransformer(EMBED_MODEL)
    return collection, embedder


@st.cache_resource(show_spinner="Connecting to Groq...")
def load_groq(api_key: str):
    """Initialize Groq client."""
    from groq import Groq
    return Groq(api_key=api_key)


# ── RAG pipeline ──────────────────────────────────────────────────────────────
def retrieve(query: str, collection, embedder) -> tuple:
    """Embed query → retrieve top-K relevant chunks from ChromaDB."""
    q_emb = embedder.encode(
        [query], normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return chunks, metadatas, distances


def build_prompt(query: str, chunks: list, history: list) -> str:
    """Assemble the full prompt: system + history + context + question."""
    # Chat history (last N messages)
    history_str = ""
    if history:
        history_str = "\n--- Previous conversation ---\n"
        for msg in history[-MAX_HISTORY:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
        history_str += "--- End of history ---\n"

    # Retrieved context
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Source {i}]\n{chunk}")
    context_str = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

{history_str}
--- Relevant knowledge retrieved from database ---
{context_str}
--- End of retrieved knowledge ---

User question: {query}

Answer:"""
    return prompt


def generate_answer(groq_client, prompt: str) -> str:
    """Call Groq API and return the text response."""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        err = str(e)
        if "401" in err or "403" in err or "api_key" in err.lower():
            return "**API key error.** Please check your GROQ_API_KEY in the `.env` file."
        return f"**Generation error:** {err}"


def source_label(filename: str) -> str:
    """Convert filename to readable source label."""
    return (
        filename
        .replace(".txt", "")
        .replace("_", " ")
        .title()
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(collection):
    with st.sidebar:
        st.header("🍄 Mushroom Expert")
        st.caption("Powered by RAG + Google Gemini")
        st.divider()

        try:
            n_chunks = collection.count()
            st.metric("Knowledge chunks", n_chunks)
        except Exception:
            st.warning("Knowledge base not loaded.")

        st.metric("AI Model", "Llama 3.3 70B (Groq)")
        st.metric("Retrieval", f"Top {TOP_K} chunks")
        st.divider()

        st.markdown("**Topics covered:**")
        topics = [
            "🦠 Mushroom diseases & pathogens",
            "🍄 Species identification",
            "🌱 Oyster mushroom cultivation",
            "💊 Disease treatment & prevention",
            "🔬 Trichoderma, Aspergillus, Rhizopus",
            "🧬 Medicinal properties",
            "⚠️ Poisonous species warnings",
        ]
        for t in topics:
            st.caption(t)

        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Data sources: TNAU, AHDB, PubMed, mushroominfo.org, expert knowledge base")


# ── Example questions ─────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What causes green mold in oyster mushroom bags?",
    "How do I treat Trichoderma contamination?",
    "What are the signs of a healthy substrate bag?",
    "Difference between single infected and mixed infected bags?",
    "What temperature is best for oyster mushroom fruiting?",
    "How dangerous is Aspergillus in mushroom cultivation?",
    "What is dry bubble disease?",
    "Which mushrooms are most poisonous?",
]


def render_examples():
    """Show clickable example questions when chat is empty."""
    st.markdown("#### Try asking:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        col = cols[i % 2]
        if col.button(q, key=f"ex_{i}", use_container_width=True):
            return q
    return None


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.title("🍄 Mushroom Expert Chat")
    st.caption(
        "Ask anything about mushroom diseases, species, cultivation, treatments, and more. "
        "Answers are grounded in a curated knowledge base."
    )

    # ── API key check ──────────────────────────────────────────────────────────
    if not GROQ_API_KEY:
        st.error(
            "**GROQ_API_KEY not found.** "
            "Add it to the `.env` file in the project folder:\n\n"
            "```\nGROQ_API_KEY=your_key_here\n```"
        )
        st.stop()

    # ── Load resources ─────────────────────────────────────────────────────────
    try:
        collection, embedder = load_retriever()
    except Exception as e:
        st.error(
            f"**Knowledge base not found.** Run the setup first:\n\n"
            f"```\npython phase6_scrape.py\npython phase6_index.py\n```\n\n"
            f"Error: {e}"
        )
        st.stop()

    try:
        groq_client = load_groq(GROQ_API_KEY)
    except Exception as e:
        st.error(f"**Groq connection failed:** {e}")
        st.stop()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    render_sidebar(collection)

    # ── Chat state ─────────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Welcome message ────────────────────────────────────────────────────────
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Hello! I'm your **mushroom expert assistant**. I can answer questions about:\n\n"
                "- **Diseases**: Trichoderma, Aspergillus, wet/dry bubble, bacterial blotch and more\n"
                "- **Species**: edible, medicinal, and poisonous mushrooms\n"
                "- **Cultivation**: substrate preparation, spawn run, fruiting, disease prevention\n"
                "- **Treatments**: chemical, biological, and cultural controls\n\n"
                "What would you like to know?"
            )
        # Clickable examples
        clicked = render_examples()
        if clicked:
            st.session_state.messages.append({"role": "user", "content": clicked})
            st.rerun()
        return

    # ── Display chat history ───────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Sources used", expanded=False):
                    for src in msg["sources"]:
                        st.caption(f"• {source_label(src)}")

    # ── Chat input ─────────────────────────────────────────────────────────────
    query = st.chat_input("Ask about mushroom diseases, species, cultivation...")
    if not query:
        return

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve → generate → display
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            chunks, metadatas, distances = retrieve(query, collection, embedder)

        with st.spinner("Generating answer..."):
            prompt = build_prompt(
                query,
                chunks,
                st.session_state.messages[:-1],  # history excludes current question
            )
            answer = generate_answer(groq_client, prompt)

        st.markdown(answer)

        # Sources
        unique_sources = list(dict.fromkeys(m["source"] for m in metadatas))
        with st.expander("📚 Sources used", expanded=False):
            for src in unique_sources:
                st.caption(f"• {source_label(src)}")

    # Save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": unique_sources,
    })


if __name__ == "__main__":
    main()
