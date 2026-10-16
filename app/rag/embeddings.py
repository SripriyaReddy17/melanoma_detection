from sentence_transformers import SentenceTransformer


# Lightweight embedding model.
# Works locally and does not require an API key.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_embedding_model():
    global _model

    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded.")

    return _model


def create_embedding(text: str) -> list[float]:
    """
    Convert text into an embedding vector.
    """

    model = get_embedding_model()

    vector = model.encode(
        text,
        normalize_embeddings=True,
    )

    return vector.tolist()