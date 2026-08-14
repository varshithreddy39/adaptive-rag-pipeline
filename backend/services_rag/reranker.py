import os
from huggingface_hub import InferenceClient


client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_TOKEN")
)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_chunks(query, chunks, top_k=3):

    if not chunks:
        return []

    result = client.text_ranking(
        query=query,
        texts=chunks,
        model=MODEL_NAME
    )

    scored_chunks = [
        (chunks[item.index], item.score)
        for item in result
    ]

    return scored_chunks[:top_k]