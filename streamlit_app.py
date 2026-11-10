import os
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="My Digital Clone", page_icon="🤖")

load_dotenv()

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATA_FOLDER = "data"


@st.cache_resource
def load_rag_system():

    documents = []

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith((".md", ".txt")):
            path = os.path.join(DATA_FOLDER, filename)

            with open(path, "r", encoding="utf-8") as file:
                documents.append({
                    "source": filename,
                    "text": file.read()
                })

    chunks = []

    for document in documents:
        words = document["text"].split()

        chunk_size = 250
        overlap = 50

        for chunk_id, start in enumerate(
            range(0, len(words), chunk_size - overlap)
        ):
            chunk_words = words[start:start + chunk_size]

            if chunk_words:
                chunks.append({
                    "source": document["source"],
                    "chunk_id": chunk_id,
                    "text": " ".join(chunk_words)
                })

    if not chunks:
        raise ValueError("No documents found in the data folder.")

    tokenized_chunks = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    embeddings_path = "embeddings.npy"

    if os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path)

        if embeddings.shape[0] != len(chunks):
            embeddings = model.encode(
                texts, convert_to_numpy=True
            )
            np.save(embeddings_path, embeddings)
    else:
        embeddings = model.encode(
            texts, convert_to_numpy=True
        )
        np.save(embeddings_path, embeddings)

    return chunks, bm25, model, embeddings


def bm25_search(query, chunks, bm25, top_k=5):

    scores = bm25.get_scores(query.lower().split())
    ranking = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "source": chunks[i]["source"],
            "chunk_id": chunks[i]["chunk_id"],
            "text": chunks[i]["text"],
            "score": round(float(scores[i]), 4)
        }
        for i in ranking
    ]


def dense_search(query, chunks, model, embeddings, top_k=5):

    query_embedding = model.encode(
        [query], convert_to_numpy=True
    )[0]

    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) *
        np.linalg.norm(query_embedding)
    )

    ranking = np.argsort(similarities)[::-1][:top_k]

    return [
        {
            "source": chunks[i]["source"],
            "chunk_id": chunks[i]["chunk_id"],
            "text": chunks[i]["text"],
            "score": round(float(similarities[i]), 4)
        }
        for i in ranking
    ]


def hybrid_search(query, chunks, bm25, model, embeddings, top_k=5):

    bm25_ranking = np.argsort(
        bm25.get_scores(query.lower().split())
    )[::-1]

    query_embedding = model.encode(
        [query], convert_to_numpy=True
    )[0]

    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) *
        np.linalg.norm(query_embedding)
    )

    dense_ranking = np.argsort(similarities)[::-1]

    rrf_scores = {}

    for rank, index in enumerate(bm25_ranking):
        rrf_scores[index] = (
            rrf_scores.get(index, 0) + 1 / (60 + rank + 1)
        )

    for rank, index in enumerate(dense_ranking):
        rrf_scores[index] = (
            rrf_scores.get(index, 0) + 1 / (60 + rank + 1)
        )

    final_ranking = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    return [
        {
            "source": chunks[i]["source"],
            "chunk_id": chunks[i]["chunk_id"],
            "text": chunks[i]["text"],
            "score": round(rrf_scores[i], 4)
        }
        for i in final_ranking[:top_k]
    ]


def generate_answer(query, retrieved_chunks):

    if not retrieved_chunks:
        return "I don't have that in my notes."

    if not GROQ_API_KEY:
        return "Groq API key was not found."

    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    prompt = f"""
You are an AI digital clone of Ali Hassan Butt.

Answer in first person as Ali.

STRICT RULES:
1. Use ONLY facts explicitly stated in the context.
2. Never guess or create personal information.
3. Do not treat a skill as a personal preference.
4. If asked about a favourite or preference,
   answer only if the context explicitly confirms it.
5. If information is missing or unclear, respond exactly:
   I don't have that in my notes.
6. Do not add unsupported details.

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI generation error: {str(e)}"


# Streamlit Interface

st.title("🤖 My Digital Clone")

st.warning("This is an AI clone, not the real person.")

try:
    chunks, bm25, model, embeddings = load_rag_system()
    st.success("Knowledge base loaded successfully!")

    st.write("Ask a question about me.")

    question = st.text_input(
        "Ask me a question",
        placeholder="Ask something about me..."
    )

    mode = st.selectbox(
        "Retrieval Mode",
        ["BM25 Only", "Dense Only", "Hybrid"]
    )

    if st.button("Search"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            if mode == "BM25 Only":
                results = bm25_search(
                    question, chunks, bm25
                )

            elif mode == "Dense Only":
                results = dense_search(
                    question, chunks, model, embeddings
                )

            else:
                results = hybrid_search(
                    question, chunks, bm25, model, embeddings
                )

            answer = generate_answer(question, results)

            st.subheader("My Answer")
            st.write(answer)

            st.subheader("Retrieved Documents")

            for result in results:
                with st.expander(
                    f"{result['source']} | Chunk {result['chunk_id']}"
                ):
                    st.write("Score:", result["score"])
                    st.write(result["text"])

except Exception as e:
    st.error(f"Application error: {str(e)}")
