import os
import numpy as np
from huggingface_hub import InferenceClient


client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_TOKEN")
)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_chunks(query, chunks, top_k=3):

    if not chunks:
        return []

    pairs = [
        [query, chunk]
        for chunk in chunks
    ]

    scores = []

    for pair in pairs:

        result = client.text_classification(
            text=pair[0],
            model=MODEL_NAME
        )

        scores.append(float(result[0]["score"]))

    scored_chunks = list(zip(chunks, scores))

    scored_chunks.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scored_chunks[:top_k]