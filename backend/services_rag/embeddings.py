from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)   
def generate_embeddings(data):

    embeddings = model.encode(
        data,
        convert_to_numpy=True
    )

    return np.array(embeddings.astype("float32"))


