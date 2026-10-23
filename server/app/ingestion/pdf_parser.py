"""Ingestion step 1: read a PDF into per-page Markdown and render the pages that contain figures."""
import re
from pathlib import Path

import pymupdf
import pymupdf4llm

from app.models import FigurePage, ParsedDocument


class PDFParser:
    # Caption lines like "Figure 2-1. ESP32 Pin Layout" or "Fig 3. Pin configuration"
    FIGURE_CAPTION = re.compile(r"^\s*Fig(?:ure|\.)?\s*\d[\w.-]*.*$", re.MULTILINE)
    MIN_IMAGE_SIZE = 150  # px; smaller embedded images are logos and icons

    def __init__(self, settings):
        self.images_dir = settings.images_dir
        self.render_dpi = settings.figure_render_dpi

    def parse(self, pdf_path: Path, file_hash: str) -> ParsedDocument:
        doc = pymupdf.open(pdf_path)

        # Human-readable name: the PDF's title metadata, else the first two lines of page 1
        label = re.sub(r"<[^>]+>", "", doc.metadata.get("title") or "").strip()
        label = (label or " ".join(doc[0].get_text().strip().splitlines()[:2]))[:100] or pdf_path.stem

        # pymupdf4llm keeps tables as real Markdown tables (rows and columns intact)
        pages_markdown = {
            page["metadata"]["page_number"]: page["text"]
            for page in pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
        }

        return ParsedDocument(
            source=pdf_path.name,
            label=label,
            file_hash=file_hash,
            pages_markdown=pages_markdown,
            figure_pages=self._render_figure_pages(doc, pdf_path.stem),
        )

    def _render_figure_pages(self, doc, file_stem: str) -> list[FigurePage]:
        """Datasheet diagrams are mostly vector drawings, not embedded pictures, so extracting images
        would miss them. Instead we render every page that has a figure caption or a large image."""
        figure_pages = []
        for page in doc:
            captions = [
                line.strip() for line in self.FIGURE_CAPTION.findall(page.get_text())
                if ". ." not in line and "...." not in line  # skip table-of-contents lines
            ]
            has_big_image = any(
                img[2] >= self.MIN_IMAGE_SIZE and img[3] >= self.MIN_IMAGE_SIZE  # img[2], img[3] = width, height
                for img in page.get_images(full=True)
            )
            if captions or has_big_image:
                image_path = self.images_dir / f"{file_stem}_p{page.number + 1}.png"
                page.get_pixmap(dpi=self.render_dpi).save(image_path)
                figure_pages.append(FigurePage(page=page.number + 1, captions=captions, image_path=image_path))
        return figure_pages
