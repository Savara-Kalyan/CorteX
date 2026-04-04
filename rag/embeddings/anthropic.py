import anthropic

from settings import settings
from rag.embeddings.base import BaseEmbeddingModel


class AnthropicEmbeddingModel(BaseEmbeddingModel):

    def __init__(self):
        self._client = anthropic.AsyncAnthropic()
        self._model = settings.embeddings.model

    async def embed_documents(self, docs: list) -> list[list[float]]:
        texts = [d.page_content if hasattr(d, "page_content") else d for d in docs]
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=[query],
        )
        return response.data[0].embedding
