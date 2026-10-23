"""Ingestion step 2: split each page's Markdown into chunks small enough to embed and search well."""
import re

import tiktoken

from app.models import Chunk, ParsedDocument


class TextChunker:
    """Chunks never cross pages, so every answer can cite an exact page. Paragraphs and tables are
    kept whole when possible; a table too big for one chunk repeats its header row in every piece,
    so a value like "3.3 V" is never separated from the column it belongs to."""

    def __init__(self, settings):
        self.max_tokens = settings.chunk_max_tokens
        self.min_tokens = settings.chunk_min_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks = []
        for page, markdown in document.pages_markdown.items():
            for text in self._split_page(markdown):
                chunks.append(Chunk(
                    # The "[label - page N]" prefix makes every chunk self-describing, so an
                    # "STM32 ADC" search doesn't match ADC text from the ESP32 datasheet.
                    text=f"[{document.label} - page {page}]\n{text}",
                    source=document.source,
                    label=document.label,
                    page=page,
                    kind="text",
                    file_hash=document.file_hash,
                ))
        return chunks

    def _split_page(self, markdown: str) -> list[str]:
        """Pack the page's blocks (paragraphs, tables - separated by blank lines) into chunks of up to max_tokens."""
        blocks = [block.strip() for block in re.split(r"\n\s*\n", markdown) if block.strip()]
        pieces, current = [], ""
        for block in blocks:
            for piece in self._split_big_block(block):
                if current and self._tokens(current + "\n\n" + piece) > self.max_tokens:
                    pieces.append(current)
                    current = piece
                else:
                    current = f"{current}\n\n{piece}" if current else piece
        if current:
            pieces.append(current)
        return [piece for piece in pieces if self._tokens(piece) >= self.min_tokens]

    def _split_big_block(self, block: str) -> list[str]:
        """Split a single block that is bigger than max_tokens by lines (table rows)."""
        if self._tokens(block) <= self.max_tokens:
            return [block]
        lines = block.splitlines()
        # A Markdown table starts with "|" and its second line is the |---|---| separator
        is_table = len(lines) > 2 and lines[0].startswith("|") and set(lines[1].replace("|", "").strip()) <= set("-: ")
        header = lines[:2] if is_table else []

        pieces, current = [], []
        for line in lines[len(header):]:
            if current and self._tokens("\n".join(header + current + [line])) > self.max_tokens:
                pieces.append("\n".join(header + current))
                current = []
            current.append(line)
        if current:
            pieces.append("\n".join(header + current))
        return pieces

    def _tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
