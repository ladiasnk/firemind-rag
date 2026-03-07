"""
query.py — The "brain" of our RAG system.

What happens when you ask a question:
  1. Your question gets converted into an embedding (same model as ingest.py)
  2. That embedding is compared against all stored chunk embeddings in ChromaDB
  3. The most similar chunks are retrieved (these are our "context")
  4. The question + retrieved context are sent to the LLM
  5. The LLM generates an answer grounded in the retrieved text

This is the core RAG loop: Retrieve → Augment → Generate
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import httpx
from openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "firemind"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"       # cheap and capable — good for RAG use cases
TOP_K = 4                        # how many chunks to retrieve per query

# The system prompt tells the LLM how to behave.
# "Only use the context provided" is key — it prevents hallucination
# by forcing the model to stay grounded in retrieved documents.
SYSTEM_PROMPT = """You are FireMind, a knowledgeable assistant specializing in wildfire science, 
behavior, prevention, and climate.

Answer questions using ONLY the context provided below. If the context doesn't contain 
enough information to answer the question, say so clearly — don't make things up.

When relevant, mention which document your information comes from.

Keep answers clear and informative. Use bullet points when listing multiple items."""


# ── RAG Engine ────────────────────────────────────────────────────────────────

class FireMindRAG:
    """
    Wraps the full RAG pipeline in a reusable class.
    Instantiate once, call .ask() many times.
    """

    def __init__(self):
        # Load the same embedding model used during ingestion
        # (Embeddings only work for retrieval if both query and docs
        #  use the exact same model — a common gotcha)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Connect to the persisted ChromaDB
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_collection(COLLECTION_NAME)

        # OpenAI client for text generation
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")
        
        # For corporate proxies with self-signed certs, disable SSL verification, therefore use httpx client with verify=False and pass it to OpenAI client
        http_client = httpx.Client(verify=False)
        self.llm = OpenAI(api_key=api_key, base_url=api_base, http_client=http_client)

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """
        Convert the question to an embedding and find the closest chunks.

        Returns a list of dicts with 'text' and 'source' for each retrieved chunk.
        """
        # Embed the question
        query_embedding = self.embedder.encode([question])[0]

        # Query ChromaDB — it returns the top_k most similar chunks
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "similarity": round(1 - dist, 3)   # cosine distance → similarity score
            })

        return chunks

    def ask(self, question: str) -> dict:
        """
        Full RAG pipeline: question → retrieve → generate → return.

        Returns a dict with:
          - answer: the LLM's response
          - sources: list of source filenames used
          - chunks: the raw retrieved chunks (useful for debugging)
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.retrieve(question)

        # Step 2: Format context block for the LLM prompt
        context_block = ""
        for i, chunk in enumerate(chunks, 1):
            context_block += f"\n--- Chunk {i} (from {chunk['source']}, similarity: {chunk['similarity']}) ---\n"
            context_block += chunk["text"] + "\n"

        # Step 3: Build the prompt
        # We explicitly separate "context" from "question" so the model
        # knows what it's allowed to use vs. what it needs to answer.
        user_message = f"""CONTEXT:
{context_block}

QUESTION: {question}

Answer based only on the context above."""

        # Step 4: Generate the answer
        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,    # low temperature = more factual, less creative
            max_tokens=600
        )

        answer = response.choices[0].message.content
        sources = list(set(c["source"] for c in chunks))

        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunks       # pass these through so the UI can show them
        }


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick sanity check — run this directly to test without the UI:
      python query.py
    """
    print("🔥 FireMind RAG — Quick Test\n")
    rag = FireMindRAG()

    test_questions = [
        "What causes most wildfires?",
        "How does wind affect fire behavior?",
        "What is defensible space?"
    ]

    for q in test_questions:
        print(f"Q: {q}")
        result = rag.ask(q)
        print(f"A: {result['answer'][:300]}...")
        print(f"   Sources: {result['sources']}\n")
