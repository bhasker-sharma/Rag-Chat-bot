"""Ingestion step 3: a vision model describes each figure page, turning diagrams into searchable text."""
from concurrent.futures import ThreadPoolExecutor

from app.models import Chunk, FigurePage, ParsedDocument


class FigureDescriber:
    NO_FIGURE = "NO_FIGURE"
    PROMPT = """This is page {page} of the electronics datasheet "{label}".
Figure captions found on this page: {captions}

Describe every figure on this page (pin layouts, block diagrams, schematics / application circuits,
timing diagrams, package drawings, graphs) so that an engineer can find it by searching and
understand it without seeing it. Include:
- the figure type and title
- every pin name and pin number
- component names and values
- signal names, connections and directions
- for graphs: axis labels, units and key values
- any notes printed with the figure
Describe only figures, not ordinary text paragraphs or tables.
If the page contains no figure at all, reply with exactly: NO_FIGURE"""

    def __init__(self, settings, openai_service):
        self.openai = openai_service
        self.workers = settings.vision_workers

    def describe(self, document: ParsedDocument) -> list[Chunk]:
        def describe_one(figure: FigurePage) -> str:
            prompt = self.PROMPT.format(page=figure.page, label=document.label, captions="; ".join(figure.captions) or "none")
            return self.openai.describe_image(figure.image_path, prompt)

        # Vision calls are slow (several seconds each), so several pages are described in parallel
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            descriptions = list(pool.map(describe_one, document.figure_pages))

        chunks = []
        for figure, description in zip(document.figure_pages, descriptions):
            if description.strip() == self.NO_FIGURE:
                figure.image_path.unlink(missing_ok=True)  # false alarm: the page only had text/tables
                continue
            chunks.append(Chunk(
                text=f"[{document.label} - page {figure.page}]\n{' '.join(figure.captions)}\n{description}",
                source=document.source,
                label=document.label,
                page=figure.page,
                kind="figure",
                file_hash=document.file_hash,
                image_file=figure.image_path.name,
            ))
        return chunks
