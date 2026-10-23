"""All backend settings in one place. Secrets and model names come from server/.env (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv


class Settings:
    def __init__(self):
        self.server_dir = Path(__file__).resolve().parent.parent
        load_dotenv(self.server_dir / ".env")

        # --- OpenAI ---
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
        self.vision_model = os.getenv("VISION_MODEL", "gpt-4.1-mini")

        # --- Folders ---
        self.data_dir = self.server_dir / "data"            # datasheet PDFs go here
        self.storage_dir = self.server_dir / "storage"      # everything generated (safe to delete & rebuild)
        self.chroma_dir = self.storage_dir / "chroma"       # vector database
        self.images_dir = self.storage_dir / "images"       # rendered figure pages
        self.sqlite_path = self.storage_dir / "chats.db"    # chat history

        # --- Ingestion tuning ---
        self.chunk_max_tokens = 600      # a few paragraphs or one table section
        self.chunk_min_tokens = 20       # smaller chunks are page footers / leftovers -> skipped
        self.figure_render_dpi = 150     # sharp enough for the vision model to read pin names
        self.vision_workers = 8          # figure pages described in parallel
        self.embedding_batch_size = 100  # chunks per embeddings API call

        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing - add it to server/.env")
        for folder in (self.storage_dir, self.chroma_dir, self.images_dir):
            folder.mkdir(parents=True, exist_ok=True)
