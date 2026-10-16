from app.rag.rag_service import retrieve_melanoma_evidence


classification = {
    "predicted_class": "melanoma",
    "melanoma_probability": 0.85,
}

abcde = {
    "asymmetry": 0.7,
    "border_irregularity": 0.6,
    "color": {
        "index": 0.5
    },
    "diameter": {
        "max_diameter_mm": 7.0
    },
}

evolution = {
    "change_detected": True,
}


result = retrieve_melanoma_evidence(
    classification=classification,
    abcde=abcde,
    evolution=evolution,
    top_k=3,
)


print("\n================ RAG QUERY ================\n")
print(result["query"])

print("\n================ RETRIEVED DOCUMENTS ================\n")

for index, document in enumerate(result["documents"], start=1):

    print(f"\n--- Document {index} ---")
    print("Source:", document["source"])
    print("Chunk:", document["chunk_index"])
    print("Similarity:", document["score"])
    print("Text:")
    print(document["text"])