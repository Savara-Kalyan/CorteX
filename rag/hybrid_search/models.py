from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    id: str
    content: str
    source_file: str | None = None
    page_number: int | None = None
    metadata: dict = Field(default_factory=dict)
    rrf_score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None


class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, description="RRF constant — higher = less rank-sensitive")
    fetch_k: int = Field(default=20, ge=1, description="Candidates fetched from each searcher before fusion")


class HybridSearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_vector_candidates: int
    total_bm25_candidates: int
