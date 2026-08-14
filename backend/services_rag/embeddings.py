from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def generate_embeddings(data):

    print("🔥 ENTERED generate_embeddings", flush=True)

    embeddings = model.encode(
        data,
        convert_to_numpy=True
    )

    print("🔥 EMBEDDINGS GENERATED", flush=True)

    return np.array(embeddings.astype("float32"))