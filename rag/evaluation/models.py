"""
Pydantic data models for the RAG evaluation framework.

Hierarchy:
  GoldenItem          — one (question, ground_truth, relevant_doc_ids) triple
  RetrievalMetrics    — P@K / Recall@K / MRR for a single query
  GenerationMetrics   — Faithfulness / AnswerRelevancy / ContextPrecision / ContextRecall
  ItemEvaluationResult— full evaluation result for one golden item
  CategoryReport      — aggregated metrics for one category
  EvaluationReport    — complete evaluation run (all categories + overall)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QueryCategory(str, Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    ADVERSARIAL = "adversarial"


# ---------------------------------------------------------------------------
# Input / corpus models
# ---------------------------------------------------------------------------


class GoldenItem(BaseModel):
    """A single entry in the golden evaluation dataset."""

    question: str
    ground_truth: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    category: QueryCategory

    model_config = {"use_enum_values": True}


class Document(BaseModel):
    """A document in the retrieval corpus."""

    id: str
    content: str
    title: str = ""
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-query metric models
# ---------------------------------------------------------------------------


class RetrievalMetrics(BaseModel):
    """Retrieval quality metrics for a single query."""

    question: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    precision_at_k: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    k: int


class GenerationMetrics(BaseModel):
    """Generation quality metrics for a single query."""

    question: str
    faithfulness: float = Field(ge=0.0, le=1.0, description="Fraction of answer claims supported by context")
    answer_relevancy: float = Field(ge=0.0, le=1.0, description="How well the answer addresses the question")
    context_precision: float = Field(ge=0.0, le=1.0, description="Fraction of retrieved chunks that are useful")
    context_recall: float = Field(ge=0.0, le=1.0, description="Whether context contains all info needed for a complete answer")


# ---------------------------------------------------------------------------
# Combined per-item result
# ---------------------------------------------------------------------------


class ItemEvaluationResult(BaseModel):
    """Full evaluation result for one golden item."""

    golden_item: GoldenItem
    retrieval: RetrievalMetrics
    generation: GenerationMetrics | None = None  # None when skipped (e.g. adversarial)
    adversarial_correctly_rejected: bool | None = None  # set for adversarial items


# ---------------------------------------------------------------------------
# Aggregate report models
# ---------------------------------------------------------------------------


class AggregateMetrics(BaseModel):
    """Averaged metrics across a set of evaluation items."""

    precision_at_k: float
    recall_at_k: float
    mrr: float
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    item_count: int


class CategoryReport(BaseModel):
    """Aggregated metrics broken down by query category."""

    category: str
    metrics: AggregateMetrics
    passed_thresholds: bool


class PassFailDecision(BaseModel):
    """Pass/fail verdict for the full evaluation run."""

    passed: bool
    verdict: Literal["READY_TO_SHIP", "DO_NOT_SHIP"]
    failing_metrics: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Complete evaluation run output."""

    overall: AggregateMetrics
    per_category: list[CategoryReport]
    adversarial_rejection_rate: float | None = None
    item_results: list[ItemEvaluationResult]
    decision: PassFailDecision
    k: int
