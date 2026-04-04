from abc import ABC, abstractmethod


class BaseVectorStore(ABC):

    @abstractmethod
    async def insert(self, docs) -> None: ...

    @abstractmethod
    async def search(self, query_embedding: list[float], top_k: int = 5) -> list: ...
