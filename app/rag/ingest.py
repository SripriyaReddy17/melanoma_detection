from pathlib import Path

from pypdf import PdfReader

from app.database import rag_documents_collection
from app.rag.embeddings import create_embedding


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def split_text(text: str) -> list[str]:

    text = " ".join(text.split())

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def extract_pdf_text(pdf_path: Path) -> str:

    reader = PdfReader(str(pdf_path))

    pages = []

    for page in reader.pages:

        text = page.extract_text() or ""

        pages.append(text)

    return "\n".join(pages)


def ingest_pdf(pdf_path: str):

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    text = extract_pdf_text(pdf_path)

    chunks = split_text(text)

    documents = []

    for index, chunk in enumerate(chunks):

        embedding = create_embedding(chunk)

        documents.append(
            {
                "source": pdf_path.name,
                "chunk_index": index,
                "text": chunk,
                "embedding": embedding,
            }
        )

    if documents:

        rag_documents_collection.delete_many(
            {
                "source": pdf_path.name
            }
        )

        rag_documents_collection.insert_many(
            documents
        )

    return {
        "source": pdf_path.name,
        "chunks": len(documents),
    }


def ingest_directory(directory: str):

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    results = []

    for pdf_file in directory_path.glob("*.pdf"):

        results.append(
            ingest_pdf(str(pdf_file))
        )

    return results
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    medical_docs_path = project_root / "data" / "medical_docs"

    print(f"Starting RAG ingestion from: {medical_docs_path}")

    results = ingest_directory(str(medical_docs_path))

    print("\nRAG ingestion completed.")
    print(results)