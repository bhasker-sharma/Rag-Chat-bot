"""The only class that talks to OpenAI: embeddings, figure descriptions and (later) chat answers."""
import base64
from pathlib import Path

from openai import OpenAI


class OpenAIService:
    def __init__(self, settings):
        self.client = OpenAI(api_key=settings.openai_api_key, max_retries=5)  # retries rate limits / network errors
        self.embedding_model = settings.embedding_model
        self.vision_model = settings.vision_model
        self.embedding_batch_size = settings.embedding_batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Turn texts into embedding vectors, sending them in batches."""
        vectors = []
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = texts[start:start + self.embedding_batch_size]
            response = self.client.embeddings.create(model=self.embedding_model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def describe_image(self, image_path: Path, prompt: str) -> str:
        """Send one image + instructions to the vision model and return its text answer."""
        image_b64 = base64.b64encode(image_path.read_bytes()).decode()
        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }],
        )
        return response.choices[0].message.content or ""
