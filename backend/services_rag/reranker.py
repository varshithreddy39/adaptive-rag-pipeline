import os
import requests


HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

API_URL = (
    f"https://router.huggingface.co/hf-inference/"
    f"models/{MODEL_NAME}"
)


def rerank_chunks(query, chunks, top_k=3):

    if not chunks:
        return []

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": {
            "query": query,
            "texts": chunks
        }
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    results = response.json()

    scored_chunks = [
        (chunks[item["index"]], item["score"])
        for item in results
    ]

    scored_chunks.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scored_chunks[:top_k]