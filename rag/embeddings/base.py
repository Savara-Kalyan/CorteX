from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):

    @abstractmethod
    async def embed_documents(self, docs: list) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]: ...
