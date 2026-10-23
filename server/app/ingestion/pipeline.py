"""The ingestion pipeline: every PDF in data/ goes through

    1. parse    PDF -> page Markdown + figure page images   (PDFParser)
    2. chunk    page Markdown -> text chunks                (TextChunker)
    3. describe figure images -> figure chunks              (FigureDescriber)
    4. embed    chunks -> vectors                           (OpenAIService)
    5. store    chunks + vectors -> ChromaDB                (VectorStore)

PDFs that are already stored and unchanged are skipped, so re-running only processes new or
changed files. PDFs deleted from data/ are removed from the store too."""
import hashlib
import logging
import time
from pathlib import Path

from app.ingestion.figure_describer import FigureDescriber
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.text_chunker import TextChunker

log = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, settings, openai_service, vector_store):
        self.data_dir = settings.data_dir
        self.parser = PDFParser(settings)
        self.chunker = TextChunker(settings)
        self.figure_describer = FigureDescriber(settings, openai_service)
        self.openai = openai_service
        self.vector_store = vector_store

    def run(self):
        pdf_paths = sorted(self.data_dir.glob("*.pdf"))
        log.info("Found %d PDFs in %s", len(pdf_paths), self.data_dir)

        for pdf_path in pdf_paths:
            file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            if self.vector_store.stored_hash(pdf_path.name) == file_hash:
                log.info("Skipping %s (already ingested, unchanged)", pdf_path.name)
                continue
            self._ingest(pdf_path, file_hash)

        for source in self.vector_store.stored_sources() - {path.name for path in pdf_paths}:
            log.info("Removing %s from the store (PDF no longer in data/)", source)
            self.vector_store.delete_document(source)

        log.info("Ingestion finished. Vector store holds %d chunks.", self.vector_store.collection.count())

    def _ingest(self, pdf_path: Path, file_hash: str):
        started = time.perf_counter()
        log.info("=== %s ===", pdf_path.name)

        log.info("[1/5] Parsing PDF")
        document = self.parser.parse(pdf_path, file_hash)
        log.info("      '%s': %d pages, %d figure pages", document.label, len(document.pages_markdown), len(document.figure_pages))

        log.info("[2/5] Chunking text")
        text_chunks = self.chunker.chunk(document)

        log.info("[3/5] Describing %d figure pages with the vision model", len(document.figure_pages))
        figure_chunks = self.figure_describer.describe(document)

        chunks = text_chunks + figure_chunks
        log.info("[4/5] Embedding %d chunks (%d text + %d figure)", len(chunks), len(text_chunks), len(figure_chunks))
        embeddings = self.openai.embed([chunk.text for chunk in chunks])

        log.info("[5/5] Storing in vector database")
        self.vector_store.replace_document(document.source, chunks, embeddings)
        log.info("Done in %.0f s", time.perf_counter() - started)
