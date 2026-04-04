from langchain_openai import OpenAIEmbeddings

from settings import settings
from rag.embeddings.base import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):

    def __init__(self):
        self._model = OpenAIEmbeddings(model=settings.embeddings.model)

    async def embed_documents(self, docs: list) -> list[list[float]]:
        texts = [d.page_content if hasattr(d, "page_content") else d for d in docs]
        return await self._model.aembed_documents(texts)

    async def embed_query(self, query: str) -> list[float]:
        return await self._model.aembed_query(query)
