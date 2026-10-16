import math

from app.database import rag_documents_collection
from app.rag.embeddings import create_embedding


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:

    if not vector_a or not vector_b:
        return 0.0

    if len(vector_a) != len(vector_b):
        return 0.0

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


def retrieve_documents(
    query: str,
    top_k: int = 3,
):

    query_embedding = create_embedding(query)

    documents = list(
        rag_documents_collection.find(
            {},
            {
                "_id": 0,
                "source": 1,
                "chunk_index": 1,
                "text": 1,
                "embedding": 1,
            },
        )
    )

    scored_documents = []

    for document in documents:

        score = cosine_similarity(
            query_embedding,
            document.get("embedding", []),
        )

        scored_documents.append(
            {
                "source": document.get("source"),
                "chunk_index": document.get("chunk_index"),
                "text": document.get("text"),
                "score": round(score, 4),
            }
        )

    scored_documents.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_documents[:top_k]