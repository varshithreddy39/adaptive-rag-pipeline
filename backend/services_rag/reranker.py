from sentence_transformers import CrossEncoder

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

cross_encoder = CrossEncoder(model_name)


def rerank_chunks(query, chunks, top_k=3):

    if not chunks:
        return []

    pairs = [
        [query, chunk]
        for chunk in chunks
    ]

    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(chunks, scores))

    scored_chunks.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scored_chunks[:top_k]