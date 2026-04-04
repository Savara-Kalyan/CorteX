from settings import settings
from rag.embeddings.base import BaseEmbeddingModel


class EmbeddingService:
    _instance: "EmbeddingService | None" = None

    def __new__(cls, backend: BaseEmbeddingModel | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, backend: BaseEmbeddingModel | None = None):
        if self._initialized:
            return
        self._initialized = True
        self._backend = backend or self._load_backend()

    def _load_backend(self) -> BaseEmbeddingModel:
        provider = settings.embeddings.provider.lower()

        if provider == "openai":
            from rag.embeddings.openai import OpenAIEmbeddingModel
            return OpenAIEmbeddingModel()

        if provider == "anthropic":
            from rag.embeddings.anthropic import AnthropicEmbeddingModel
            return AnthropicEmbeddingModel()

        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. Choose 'openai' or 'anthropic'."
        )

    async def embed_documents(self, docs: list) -> list[list[float]]:
        return await self._backend.embed_documents(docs)

    async def embed_query(self, query: str) -> list[float]:
        return await self._backend.embed_query(query)
