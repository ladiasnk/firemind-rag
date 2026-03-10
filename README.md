# 🔥 FireMind RAG

A Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions about wildfire science, behavior, prevention, and climate change.

Built as a clear, well-commented demonstration of the full RAG pipeline — from raw documents to a grounded, cited answer.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)

---

## What is RAG? (Beginner Explanation)

Large Language Models (LLMs) like GPT are powerful, but they have two problems:

1. **They hallucinate** — they can confidently state things that are wrong
2. **They're frozen in time** — they only know what was in their training data

**RAG solves this** by adding a retrieval step before generation:

```
Your Question
     ↓
[Embed question into numbers]
     ↓
[Search a knowledge base for similar text chunks]
     ↓
[Give those chunks to the LLM as context]
     ↓
[LLM answers using ONLY the retrieved context]
     ↓
Grounded Answer (with sources)
```

This means the LLM's answers are grounded in *your* documents, not its training data. It can cite sources. It can say "I don't know" when the documents don't cover something. This makes it reliable enough to use in real applications.

---

## The Pipeline (Step by Step)

### Phase 1: Ingestion (`ingest.py`)
Run once to build the knowledge base.

```
.md files in /data
       ↓
   [chunking]        ← split into ~500‑char overlapping chunks using
                        LangChain's RecursiveCharacterTextSplitter
       ↓
   [embedding]       ← each chunk → list of 384 numbers (semantic meaning)
       ↓
   [ChromaDB]        ← stored locally on disk, indexed for fast search
```

### Phase 2: Query (`query.py`)
Runs every time a user asks a question.

```
User question
       ↓
   [embed question]  ← same model, same vector space as ingestion
       ↓
   [vector search]   ← cosine similarity → top 4 most relevant chunks
       ↓
   [prompt assembly] ← question + retrieved chunks → LLM prompt
       ↓
   [LLM generation]  ← gpt-4o-mini generates answer from context only
       ↓
Answer + sources
```

---

## Project Structure

```
firemind-rag/
│
├── data/                        # Your knowledge base (add .md files here)
│   ├── wildfire_causes.md
│   ├── fire_behavior.md
│   ├── prevention_guide.md
│   └── climate_and_fire.md
│
├── ingest.py                    # One-time ingestion pipeline
├── query.py                     # RAG query engine
├── app.py                       # Streamlit web UI
│
├── chroma_db/                   # Auto-created by ingest.py (gitignored)
├── requirements.txt
├── .env.example                 # Copy to .env and add your OpenAI key
└── .gitignore
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourusername/firemind-rag.git
cd firemind-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` embedding model (~90MB). This only happens once.

### 2. Set your OpenAI API key

```bash

# open .env and add your key:
# OPENAI_API_KEY=sk-...
# If you have API base (connect through corporate proxy for example), add that one as well
# OPENAI_API_BASE=...
```

Get an API key at [platform.openai.com](https://platform.openai.com). The app uses `gpt-4o-mini` which costs ~$0.15 per million input tokens — a typical Q&A session costs fractions of a cent.

### 3. Ingest the documents

```bash
python ingest.py
```

You should see something like:
```
[1/4] Loading documents from /data ...
  📄 Loading: wildfire_causes.md
  📄 Loading: fire_behavior.md
  ...
  ✅ Loaded 4 files → 47 chunks total

[3/4] Loading embedding model (all-MiniLM-L6-v2) ...
[4/4] Storing in ChromaDB ...
  ✅ Done! 47 chunks stored in 'chroma_db/'
```

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Example Questions to Try

- *"What causes most wildfires in the United States?"*
- *"How does wind affect how fire spreads?"*
- *"What is defensible space and why does it matter?"*
- *"How is climate change making wildfires worse?"*
- *"What should I do to prepare my home for wildfire season?"*
- *"What is the fire triangle?"*

---

## Adding Your Own Documents

Just drop `.md` files into the `/data` folder and re-run ingestion:

```bash
python ingest.py --reset
```

The `--reset` flag deletes and rebuilds the vector database from scratch. Without it, the script skips ingestion if the database already exists.

---

## Known Limitations & What to Improve Next

This is an intentionally simple implementation. Here's what a production system would improve:

| Area | Current | Better |
|---|---|---|
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Smaller, tuned chunks or a domain‑aware splitter |
| **Chunk size** | Fixed 500 chars | Tuned per document type; evaluated against retrieval quality |
| **Embeddings** | `all-MiniLM-L6-v2` (general) | Domain-fine-tuned model for fire/climate content |
| **Retrieval** | Top-K similarity only | Hybrid search (keyword + vector), re-ranking |
| **Context** | No conversation memory | Multi-turn with message history |
| **Evaluation** | Manual/eyeball | RAGAS, TruLens, or custom eval harness |
| **Documents** | 4 markdown files | Live data feeds (NIFC, NOAA, NASA FIRMS) |
| **Auth** | None | User auth for multi-tenant use |

---

## Tech Stack

| Component | Library | Why |
|---|---|---|
| Embeddings | `sentence-transformers` | Free, local, no API key needed |
| Vector database | `ChromaDB` | Simple local setup, no server required |
| LLM | `openai` (gpt-4o-mini) | Cheap, capable, reliable |
| UI | `streamlit` | Fast to build, great for data apps |
| Env vars | `python-dotenv` | Standard practice for secrets |

---

## How the Retrieval Works (The Math, Simply)

When you type a question, it gets converted into a vector — a list of 384 numbers. Each number represents some dimension of meaning (not literally interpretable, but think of it like coordinates in "meaning space").

Each stored chunk also has a vector. **Cosine similarity** measures the angle between two vectors — a score of 1.0 means identical direction (very relevant), 0.0 means perpendicular (unrelated).

ChromaDB does this comparison fast using an algorithm called HNSW (Hierarchical Navigable Small World graph), which avoids comparing your query against every single chunk.

---

## License

MIT — do whatever you want with it.
