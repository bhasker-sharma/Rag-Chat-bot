"""ChromaDB collection holding every chunk with its embedding.
Ingestion writes to it; retrieval (next step) searches it."""
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.models import Chunk


class VectorStore:
    COLLECTION_NAME = "datasheets"
    ADD_BATCH_SIZE = 500  # Chroma limits how many rows one add() call may contain

    def __init__(self, settings):
        client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection(self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    def stored_hash(self, source: str) -> str | None:
        """File hash of the stored version of this PDF, or None if it was never ingested."""
        found = self.collection.get(where={"source": source}, limit=1, include=["metadatas"])
        return found["metadatas"][0]["file_hash"] if found["ids"] else None

    def stored_sources(self) -> set[str]:
        """File names of all PDFs currently in the store."""
        return {meta["source"] for meta in self.collection.get(include=["metadatas"])["metadatas"]}

    def delete_document(self, source: str):
        self.collection.delete(where={"source": source})

    def replace_document(self, source: str, chunks: list[Chunk], embeddings: list[list[float]]):
        """Remove any older version of this PDF, then store its new chunks."""
        self.delete_document(source)
        for start in range(0, len(chunks), self.ADD_BATCH_SIZE):
            batch = chunks[start:start + self.ADD_BATCH_SIZE]
            self.collection.add(
                ids=[f"{source}::{start + i}" for i in range(len(batch))],
                documents=[chunk.text for chunk in batch],
                embeddings=embeddings[start:start + self.ADD_BATCH_SIZE],
                metadatas=[{
                    "source": chunk.source,
                    "label": chunk.label,
                    "page": chunk.page,
                    "kind": chunk.kind,
                    "image_file": chunk.image_file,
                    "file_hash": chunk.file_hash,
                } for chunk in batch],
            )
