from pathlib import Path
from dataclasses import dataclass

import yaml


@dataclass
class IngestionSettings:
    char_count_threshold: int
    sample_pages: int


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
class Settings:
    llm: LLMSettings
    embeddings: EmbeddingSettings
    vector_store: VectorStoreSettings
    ingestion: IngestionSettings
    query_understanding: QueryUnderstandingSettings


def _load_settings() -> Settings:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    vs = raw["VECTOR_STORE"]
    emb = raw["EMBEDDINGS"]
    ing = raw["INGESTION"]
    qu = raw["QUERY_UNDERSTANDING"]

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
            connection_string=vs.get("CONNECTION_STRING"),
            url=vs.get("URL"),
            api_key=vs.get("API_KEY"),
            collection_name=vs.get("COLLECTION_NAME", "documents"),
        ),
        ingestion=IngestionSettings(
            char_count_threshold=ing["CHAR_COUNT_THRESHOLD"],
            sample_pages=ing["SAMPLE_PAGES"],
        ),
        query_understanding=QueryUnderstandingSettings(
            provider=qu["PROVIDER"],
            model=qu["MODEL"],
            temperature=qu["TEMPERATURE"],
        ),
    )


settings = _load_settings()
