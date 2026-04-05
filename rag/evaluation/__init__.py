"""
RAG Evaluation Framework
========================

Modules
-------
models              — Pydantic data models (GoldenItem, EvaluationReport, …)
metrics             — Pure metric functions (precision_at_k, recall_at_k, MRR)
dataset             — GoldenDataset management + LLM-assisted generation
retrieval_evaluator — RetrievalEvaluator: P@K, Recall@K, MRR
generation_evaluator— GenerationEvaluator: Faithfulness, Relevancy (LLM judge)
service             — EvaluationService: full orchestration + pass/fail report

Quick start::

    from rag.evaluation import EvaluationService, EvaluationConfig, GoldenDataset

    dataset = GoldenDataset.from_jsonl("golden.jsonl")
    config  = EvaluationConfig(k=5, run_generation_eval=False)
    service = EvaluationService(config=config, search_fn=my_search_fn)
    report  = await service.run(dataset)
    print(report.decision.verdict)
"""

from rag.evaluation.dataset import DatasetGenerator, GoldenDataset
from rag.evaluation.generation_evaluator import GenerationEvaluator
from rag.evaluation.metrics import (
    aggregate_generation_scores,
    aggregate_retrieval_scores,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from rag.evaluation.models import (
    AggregateMetrics,
    CategoryReport,
    Document,
    EvaluationReport,
    GenerationMetrics,
    GoldenItem,
    ItemEvaluationResult,
    PassFailDecision,
    QueryCategory,
    RetrievalMetrics,
)
from rag.evaluation.retrieval_evaluator import RetrievalEvaluator
from rag.evaluation.service import EvaluationConfig, EvaluationService

__all__ = [
    # Service
    "EvaluationService",
    "EvaluationConfig",
    # Dataset
    "GoldenDataset",
    "DatasetGenerator",
    # Evaluators
    "RetrievalEvaluator",
    "GenerationEvaluator",
    # Models
    "GoldenItem",
    "Document",
    "QueryCategory",
    "RetrievalMetrics",
    "GenerationMetrics",
    "AggregateMetrics",
    "CategoryReport",
    "ItemEvaluationResult",
    "EvaluationReport",
    "PassFailDecision",
    # Metric functions
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "aggregate_retrieval_scores",
    "aggregate_generation_scores",
]
