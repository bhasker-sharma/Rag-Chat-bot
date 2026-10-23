"""Data objects passed between the pipeline steps."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FigurePage:
    page: int             # 1-based, same as in a PDF viewer
    captions: list[str]   # e.g. ["Figure 2-1. ESP32 Pin Layout (QFN 6*6, Top View)"]
    image_path: Path      # the page rendered as PNG


@dataclass
class ParsedDocument:
    source: str                     # PDF file name, e.g. "esp32_datasheet_en.pdf"
    label: str                      # human-readable name, e.g. "ESP32 Series Datasheet Version 5.3"
    file_hash: str                  # changes when the PDF is replaced by a new version
    pages_markdown: dict[int, str]  # page number -> page content as Markdown (tables included)
    figure_pages: list[FigurePage]


@dataclass
class Chunk:
    """One searchable piece of a datasheet: a section of text, or the description of a figure page."""
    text: str
    source: str
    label: str
    page: int
    kind: str             # "text" or "figure"
    file_hash: str
    image_file: str = ""  # PNG file name in storage/images for figure chunks, "" for text chunks
