import os
from pathlib import Path
from dataclasses import dataclass

import yaml


@dataclass
class DomainChunkingSettings:
    chunk_size: int
    chunk_overlap: int


@dataclass
class ChunkingSettings:
    chunk_size: int
    chunk_overlap: int
    domain_overrides: dict[str, DomainChunkingSettings] = None

    def for_domain(self, domain: str) -> DomainChunkingSettings:
        if self.domain_overrides and domain in self.domain_overrides:
            return self.domain_overrides[domain]
        return DomainChunkingSettings(self.chunk_size, self.chunk_overlap)



@dataclass
class LLMSettings:
    provider: str
    model: str


@dataclass
class EmbeddingSettings:
    provider: str
    model: str


@dataclass
class QueryUnderstandingSettings:
    provider: str
    model: str
    temperature: float


@dataclass
class VectorStoreSettings:
    provider: str
    # pgvector
    connection_string: str | None = None
    # qdrant
    url: str | None = None
    api_key: str | None = None
    collection_name: str = "documents"


@dataclass
class RetrievalSettings:
    top_k: int
    fetch_k: int


@dataclass
class Settings:
    llm: LLMSettings
    embeddings: EmbeddingSettings
    vector_store: VectorStoreSettings
    chunking: ChunkingSettings
    query_understanding: QueryUnderstandingSettings
    retrieval: RetrievalSettings


def _load_settings() -> Settings:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    vs = raw["VECTOR_STORE"]
    emb = raw["EMBEDDINGS"]
    qu = raw["QUERY_UNDERSTANDING"]
    ch = raw["CHUNKING"]
    ret = raw.get("RETRIEVAL", {})

    return Settings(
        llm=LLMSettings(
            provider=raw["LLM"]["PROVIDER"],
            model=raw["LLM"]["MODEL"],
        ),
        embeddings=EmbeddingSettings(
            provider=emb["PROVIDER"],
            model=emb["MODEL"],
        ),
        vector_store=VectorStoreSettings(
            provider=vs["PROVIDER"],
            connection_string=os.environ.get("PGVECTOR_URL") or vs.get("CONNECTION_STRING"),
            url=os.environ.get("QDRANT_URL") or vs.get("URL"),
            api_key=os.environ.get("QDRANT_API_KEY") or vs.get("API_KEY"),
            collection_name=vs.get("COLLECTION_NAME", "documents"),
        ),
        chunking=ChunkingSettings(
            chunk_size=ch["CHUNK_SIZE"],
            chunk_overlap=ch["CHUNK_OVERLAP"],
            domain_overrides={
                domain: DomainChunkingSettings(
                    chunk_size=cfg["CHUNK_SIZE"],
                    chunk_overlap=cfg["CHUNK_OVERLAP"],
                )
                for domain, cfg in ch.get("DOMAIN_OVERRIDES", {}).items()
            },
        ),
        query_understanding=QueryUnderstandingSettings(
            provider=qu["PROVIDER"],
            model=qu["MODEL"],
            temperature=qu["TEMPERATURE"],
        ),
        retrieval=RetrievalSettings(
            top_k=ret.get("TOP_K", 5),
            fetch_k=ret.get("FETCH_K", 40),
        ),
    )


settings = _load_settings()
