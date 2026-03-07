"""
app.py — Streamlit web interface for FireMind RAG.

Run with:
  streamlit run app.py
"""

import streamlit as st
from query import FireMindRAG

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FireMind RAG",
    page_icon="🔥",
    layout="centered"
)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .source-badge {
        display: inline-block;
        background: #2d2d2d;
        color: #f0f0f0;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        margin: 2px;
    }
    .chunk-box {
        background: #1a1a2e;
        border-left: 3px solid #e25822;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #ccc;
    }
    .similarity-score {
        color: #e25822;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔥 FireMind")
st.caption("Ask anything about wildfire science, behavior, prevention, and climate.")
st.divider()

# ── Load RAG engine (cached so it only loads once) ────────────────────────────

@st.cache_resource
def load_rag():
    """
    @st.cache_resource means this runs once and is shared across sessions.
    Loading the embedding model takes ~2 seconds — we don't want to repeat that
    on every question.
    """
    try:
        return FireMindRAG(), None
    except Exception as e:
        return None, str(e)

rag, error = load_rag()

if error:
    st.error(f"⚠️ Could not load RAG engine: {error}")
    st.info("Make sure you've run `python ingest.py` and set your `OPENAI_API_KEY` in `.env`")
    st.stop()

# ── Suggested questions ───────────────────────────────────────────────────────

st.markdown("**Try asking:**")
example_questions = [
    "What causes most wildfires in the US?",
    "How does wind affect how a fire spreads?",
    "What is defensible space and why does it matter?",
    "How is climate change making wildfires worse?",
    "What should I pack in an evacuation go-bag?",
]

# Show clickable example buttons in a row
cols = st.columns(len(example_questions))
selected_example = None
for col, question in zip(cols, example_questions):
    if col.button(question[:35] + "…", use_container_width=True):
        selected_example = question

st.divider()

# ── Main question input ───────────────────────────────────────────────────────

# Pre-fill with the clicked example, or let the user type
default_value = selected_example if selected_example else ""
question = st.text_input(
    "Your question:",
    value=default_value,
    placeholder="e.g. What makes a red flag warning day dangerous?",
    label_visibility="collapsed"
)

ask_button = st.button("Ask FireMind 🔥", type="primary", use_container_width=True)

# ── Answer ────────────────────────────────────────────────────────────────────

if ask_button and question.strip():

    with st.spinner("Searching knowledge base..."):
        result = rag.ask(question)

    st.markdown("### Answer")
    st.markdown(result["answer"])

    # Show which source documents were used
    st.markdown("**Sources consulted:**")
    for src in result["sources"]:
        st.markdown(f'<span class="source-badge">📄 {src}</span>', unsafe_allow_html=True)

    # Expandable section showing the raw retrieved chunks
    # This is great for demonstrating/debugging RAG — you can see exactly
    # what context the model received before generating its answer.
    with st.expander("🔍 Show retrieved chunks (RAG transparency)"):
        st.caption(
            "These are the exact text chunks retrieved from the knowledge base "
            "before the answer was generated. The LLM only 'sees' these chunks — "
            "nothing else."
        )
        for i, chunk in enumerate(result["chunks"], 1):
            st.markdown(
                f'<div class="chunk-box">'
                f'<strong>Chunk {i}</strong> — {chunk["source"]} '
                f'<span class="similarity-score">(similarity: {chunk["similarity"]})</span>'
                f'<br><br>{chunk["text"][:400]}{"…" if len(chunk["text"]) > 400 else ""}'
                f'</div>',
                unsafe_allow_html=True
            )

elif ask_button and not question.strip():
    st.warning("Please enter a question first.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About FireMind")
    st.markdown("""
    **FireMind** is a RAG (Retrieval-Augmented Generation) demo.

    Instead of relying on the LLM's training data, it:
    1. Searches a local knowledge base of wildfire documents
    2. Retrieves the most relevant passages
    3. Asks the LLM to answer using *only* those passages

    This keeps answers grounded in real sources and makes the system auditable.

    ---
    **Knowledge base covers:**
    - 🔥 Wildfire causes & ignition
    - 💨 Fire behavior & spread
    - 🏠 Prevention & home hardening
    - 🌡️ Climate change & fire

    ---
    **Stack:**
    - Embeddings: `sentence-transformers`
    - Vector DB: `ChromaDB`
    - LLM: `gpt-4o-mini`
    - UI: `Streamlit`
    """)

    st.divider()
    try:
        count = rag.collection.count()
        st.metric("Chunks in knowledge base", count)
    except Exception:
        pass
