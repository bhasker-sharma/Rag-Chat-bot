"""Backend entry point - the only file you run.

    python main.py ingest    # add new / changed PDFs from data/ to the vector database
"""
import argparse
import logging

from app.config import Settings
from app.ingestion.pipeline import IngestionPipeline
from app.services.openai_service import OpenAIService
from app.storage.vector_store import VectorStore


class Application:
    """Creates the shared services once and hands them to whichever pipeline is run.
    New pipelines (retrieval / chat API) are added here as new commands."""

    COMMANDS = {
        "ingest": "add new / changed PDFs from data/ to the vector database",
    }

    def __init__(self):
        self.settings = Settings()
        self.openai_service = OpenAIService(self.settings)
        self.vector_store = VectorStore(self.settings)

    def ingest(self):
        IngestionPipeline(self.settings, self.openai_service, self.vector_store).run()


if __name__ == "__main__":
    # Libraries only log warnings/errors; our own code (the "app" package) logs its progress too
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("app").setLevel(logging.INFO)

    arg_parser = argparse.ArgumentParser(description="Electronics datasheet RAG backend")
    arg_parser.add_argument("command", choices=Application.COMMANDS,
                            help="; ".join(f"{name}: {text}" for name, text in Application.COMMANDS.items()))
    command = arg_parser.parse_args().command

    getattr(Application(), command)()
