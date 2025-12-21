import os
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- SAFETY CHECKS ----------------

if not os.path.exists("embeddings.joblib"):
    print("❌ embeddings.joblib not found. Cannot proceed.")
    exit()

# ---------------- LOAD EMBEDDINGS ----------------

df = joblib.load("embeddings.joblib")

if df.empty:
    print("❌ embeddings.joblib is empty.")
    exit()

# ---------------- EMBEDDING FUNCTION ----------------

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    if r.status_code != 200:
        raise RuntimeError("Embedding API failed")

    return r.json()["embeddings"]

# ---------------- LLM INFERENCE ----------------

def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    if r.status_code != 200:
        raise RuntimeError("LLM generation failed")

    return r.json()["response"]

# ---------------- USER QUERY ----------------

incoming_query = input("Ask a question related to the course: ").strip()

if not incoming_query:
    print("❌ Empty question not allowed.")
    exit()

# ---------------- SIMILARITY SEARCH ----------------

question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(
    np.vstack(df["embedding"].values),
    [question_embedding]
).flatten()

# similarity threshold
if similarities.max() < 0.30:
    print("❌ Question not related to the course.")
    exit()

top_k = 5
top_indices = similarities.argsort()[::-1][:top_k]
top_chunks = df.loc[top_indices]

# ---------------- PROMPT CREATION ----------------

prompt = f"""
I am teaching web development in my Sigma web development course.
Below are subtitle chunks from the course videos:

{top_chunks[["title", "number", "start", "end", "text"]].to_json(orient="records")}

---------------------------------
User Question:
"{incoming_query}"

Answer in a human-friendly way.
Mention:
- Which video
- What topic
- Approximate timestamp

Do NOT mention JSON, embeddings, or internal processing.
If unrelated, politely refuse.
"""

with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

# ---------------- LLM RESPONSE ----------------

response = inference(prompt)

print("\n📌 Answer:\n")
print(response)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)
