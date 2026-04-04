from pydantic import BaseModel, Field, computed_field
import hashlib


class DocumentInsert(BaseModel):
    content: str
    embedding: list[float]
    source_file: str | None = None
    page_number: int | None = None
    chunk_index: int
    total_chunks: int
    access_level: str = "internal"
    created_by: str
    doc_type: str = "unknown"
    chunk_type: str = "text"
    extraction_method: str = "docling"
    extraction_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    chunk_length: int
    metadata: dict = Field(default_factory=dict)

    @computed_field
    @property
    def doc_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()
