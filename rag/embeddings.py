from abc import ABC, abstractmethod

from settings import settings


class BaseEmbeddingModel(ABC):

    @abstractmethod
    async def embed_documents(self, docs: list) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]: ...


class OpenAIEmbeddingModel(BaseEmbeddingModel):

    def __init__(self):
        from langchain_openai import OpenAIEmbeddings
        self._model = OpenAIEmbeddings(model=settings.embeddings.model)

    async def embed_documents(self, docs: list) -> list[list[float]]:
        texts = [d.page_content if hasattr(d, "page_content") else d for d in docs]
        return await self._model.aembed_documents(texts)

    async def embed_query(self, query: str) -> list[float]:
        return await self._model.aembed_query(query)


class AnthropicEmbeddingModel(BaseEmbeddingModel):

    def __init__(self):
        import anthropic
        self._client = anthropic.AsyncAnthropic()
        self._model = settings.embeddings.model

    async def embed_documents(self, docs: list) -> list[list[float]]:
        texts = [d.page_content if hasattr(d, "page_content") else d for d in docs]
        response = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        response = await self._client.embeddings.create(model=self._model, input=[query])
        return response.data[0].embedding


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
            return OpenAIEmbeddingModel()
        if provider == "anthropic":
            return AnthropicEmbeddingModel()
        raise ValueError(f"Unsupported embedding provider: '{provider}'. Choose 'openai' or 'anthropic'.")

    async def embed_documents(self, docs: list) -> list[list[float]]:
        return await self._backend.embed_documents(docs)

    async def embed_query(self, query: str) -> list[float]:
        return await self._backend.embed_query(query)
