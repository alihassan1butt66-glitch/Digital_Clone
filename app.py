
import os
import numpy as np
import gradio as gr

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Folder containing personal documents
DATA_FOLDER = "data"

# Load documents
documents = []

for filename in os.listdir(DATA_FOLDER):
    if filename.endswith((".md", ".txt")):
        path = os.path.join(DATA_FOLDER, filename)

        with open(path, "r", encoding="utf-8") as file:
            documents.append({
                "source": filename,
                "text": file.read()
            })

print("Documents loaded:", len(documents))

# Split documents into chunks
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

print("Total chunks created:", len(chunks))

if not chunks:
    raise ValueError(
        "No documents found. Please add files to the data folder."
    )

# Prepare BM25 search
tokenized_chunks = [
    chunk["text"].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

print("BM25 search system is ready!")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create or load dense embeddings
texts = [chunk["text"] for chunk in chunks]

if os.path.exists("embeddings.npy"):
    embeddings = np.load("embeddings.npy")

    if embeddings.shape[0] != len(chunks):
        print("Documents changed. Creating new embeddings...")
        embeddings = model.encode(
            texts,
            convert_to_numpy=True
        )
        np.save("embeddings.npy", embeddings)
else:
    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )
    np.save("embeddings.npy", embeddings)

print("Dense embeddings created and saved!")
print("Your document retrieval system is ready!")


# BM25 Search
def bm25_search(query, top_k=5):

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    ranking = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranking:
        results.append({
            "source": chunks[index]["source"],
            "chunk_id": chunks[index]["chunk_id"],
            "text": chunks[index]["text"],
            "score": round(float(scores[index]), 4)
        })

    return results


# Dense Search
def dense_search(query, top_k=5):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )[0]

    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) *
        np.linalg.norm(query_embedding)
    )

    ranking = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in ranking:
        results.append({
            "source": chunks[index]["source"],
            "chunk_id": chunks[index]["chunk_id"],
            "text": chunks[index]["text"],
            "score": round(float(similarities[index]), 4)
        })

    return results


# Hybrid Search using Reciprocal Rank Fusion
def hybrid_search(query, top_k=5):

    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranking = np.argsort(bm25_scores)[::-1]

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )[0]

    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) *
        np.linalg.norm(query_embedding)
    )

    dense_ranking = np.argsort(similarities)[::-1]

    # Reciprocal Rank Fusion
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

    results = []

    for index in final_ranking[:top_k]:
        results.append({
            "source": chunks[index]["source"],
            "chunk_id": chunks[index]["chunk_id"],
            "text": chunks[index]["text"],
            "score": round(rrf_scores[index], 4)
        })

    return results



# AI Answer Generation using Groq
def generate_answer(query, retrieved_chunks):

    if not retrieved_chunks:
        return "I don't have that in my notes."

    if not GROQ_API_KEY:
        return "Groq API key was not found. Please check your .env file."

    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    prompt = f"""
You are an AI digital clone of Ali Hassan Butt.

Answer in first person as Ali.

STRICT RULES:
1. Use ONLY facts explicitly stated in the context.
2. Never guess, assume, or create personal information.
3. Do not treat a skill, interest, or experience as a personal preference.
4. If asked about a favourite, opinion, or preference,
   answer only if the context explicitly confirms it.
5. If the requested information is missing or unclear,
   respond exactly:
   I don't have that in my notes.
6. Do not add unsupported explanations or details.

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
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200,
            temperature=0.1
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI generation error: {str(e)}"


# Search function for Gradio
def search_documents(query, mode):

    if not query.strip():
        return "Please enter a question.", ""

    if mode == "BM25 Only":
        results = bm25_search(query)

    elif mode == "Dense Only":
        results = dense_search(query)

    else:
        results = hybrid_search(query)

    answer = generate_answer(query, results)

    sources = ""

    for result in results:
        sources += f"\nSource: {result['source']}\n"
        sources += f"Chunk ID: {result['chunk_id']}\n"
        sources += f"Score: {result['score']}\n"
        sources += f"Content: {result['text'][:300]}\n"
        sources += "-" * 40 + "\n"

    return answer, sources


# Gradio Chatbot Interface
with gr.Blocks() as demo:

    gr.Markdown("# My Digital Clone")

    gr.Markdown(
        "⚠️ This is an AI clone, not the real person."
    )

    question = gr.Textbox(
        label="Ask me a question",
        placeholder="Ask something about me..."
    )

    mode = gr.Dropdown(
        choices=[
            "BM25 Only",
            "Dense Only",
            "Hybrid"
        ],
        value="Hybrid",
        label="Retrieval Mode"
    )

    search_button = gr.Button("Search")

    answer_output = gr.Textbox(
        label="My Answer",
        lines=5
    )

    sources_output = gr.Textbox(
        label="Retrieved Documents",
        lines=12
    )

    search_button.click(
        fn=search_documents,
        inputs=[question, mode],
        outputs=[answer_output, sources_output]
    )


if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7861))
)
