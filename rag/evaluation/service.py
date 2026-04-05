"""
EvaluationService — main orchestrator for the RAG evaluation framework.

Responsibilities:
  • Run retrieval evaluation against the golden dataset
  • Run generation evaluation (optional, requires LLM)
  • Aggregate per-category and overall metrics
  • Compute adversarial rejection rate
  • Produce a pass/fail decision against configurable thresholds
  • Return a structured EvaluationReport

Usage::

    from rag.evaluation.service import EvaluationService, EvaluationConfig
    from rag.evaluation.dataset import GoldenDataset

    dataset  = GoldenDataset.from_jsonl("golden.jsonl")
    config   = EvaluationConfig(k=5, run_generation_eval=True)
    service  = EvaluationService(config=config, search_fn=my_search_fn)

    report   = await service.run(dataset)
    print(report.decision.verdict)          # "READY_TO_SHIP" or "DO_NOT_SHIP"
    print(report.overall.precision_at_k)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from rag.evaluation.dataset import GoldenDataset
from rag.evaluation.generation_evaluator import GenerationEvaluator, GenerationEvaluationError
from rag.evaluation.metrics import aggregate_generation_scores, aggregate_retrieval_scores
from rag.evaluation.models import (
    AggregateMetrics,
    CategoryReport,
    EvaluationReport,
    GenerationMetrics,
    GoldenItem,
    ItemEvaluationResult,
    PassFailDecision,
    QueryCategory,
    RetrievalMetrics,
)
from rag.evaluation.retrieval_evaluator import RetrievalEvaluator, RetrievalEvaluationError

logger = logging.getLogger(__name__)

# Type alias
SearchFn = Callable[[str, int], Awaitable[list[str]]]
GenerationFn = Callable[[str, str], Awaitable[str]]  # (question, context) -> answer


# ---------------------------------------------------------------------------
# Default score thresholds
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "precision_at_k":  0.75,
    "recall_at_k":     0.70,
    "mrr":             0.70,
    "faithfulness":    0.80,
    "answer_relevancy": 0.80,
    "context_precision": 0.70,
    "context_recall":  0.70,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvaluationConfig:
    """
    Configuration for an evaluation run.

    Args:
        k:                     Rank cut-off for retrieval metrics (default 5).
        run_generation_eval:   Whether to run LLM-based generation evaluation
                               (requires a generation_fn to be supplied).
        thresholds:            Score thresholds for the pass/fail decision.
                               Keys must match AggregateMetrics field names.
        retrieval_concurrency: Max concurrent search calls.
        generation_concurrency: Max concurrent LLM judge calls.
    """

    k: int = 5
    run_generation_eval: bool = False
    thresholds: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_THRESHOLDS))
    retrieval_concurrency: int = 5
    generation_concurrency: int = 4


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EvaluationServiceError(Exception):
    """Base exception for evaluation service errors."""


class EvaluationConfigError(EvaluationServiceError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[EvaluationService] Invalid configuration: {reason}")


# ---------------------------------------------------------------------------
# EvaluationService
# ---------------------------------------------------------------------------


class EvaluationService:
    """
    Full RAG evaluation pipeline.

    Args:
        search_fn:      Async callable ``(query: str, top_k: int) -> list[str]``
                        returning ordered document IDs. **Required**.
        generation_fn:  Async callable ``(question: str, context: str) -> str``
                        returning the RAG system's answer.  Required only when
                        ``config.run_generation_eval=True``.
        llm_caller:     Optional LLM caller override forwarded to
                        GenerationEvaluator (uses project settings by default).
        config:         EvaluationConfig (defaults used when not supplied).
    """

    def __init__(
        self,
        search_fn: SearchFn,
        generation_fn: GenerationFn | None = None,
        llm_caller: Callable[[str], Awaitable[str]] | None = None,
        config: EvaluationConfig | None = None,
    ) -> None:
        self._config = config or EvaluationConfig()
        self._search_fn = search_fn
        self._generation_fn = generation_fn

        if self._config.run_generation_eval and generation_fn is None:
            raise EvaluationConfigError(
                "generation_fn must be provided when run_generation_eval=True."
            )

        self._retrieval_evaluator = RetrievalEvaluator(
            search_fn=search_fn,
            k=self._config.k,
            concurrency=self._config.retrieval_concurrency,
        )
        self._generation_evaluator = GenerationEvaluator(
            llm_caller=llm_caller,
            concurrency=self._config.generation_concurrency,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, dataset: GoldenDataset) -> EvaluationReport:
        """
        Execute a full evaluation run against the provided dataset.

        Steps:
          1. Evaluate retrieval metrics for all answerable items.
          2. (Optional) Evaluate generation metrics for answerable items.
          3. Compute adversarial rejection rate.
          4. Aggregate per-category and overall metrics.
          5. Apply pass/fail thresholds.

        Args:
            dataset: GoldenDataset containing golden items.

        Returns:
            EvaluationReport with all metrics, category breakdown, and verdict.

        Raises:
            EvaluationServiceError: when the dataset is empty or a critical
                                    failure prevents evaluation from completing.
        """
        if len(dataset) == 0:
            raise EvaluationServiceError("Dataset is empty — nothing to evaluate.")

        answerable = dataset.answerable_items()
        adversarial = dataset.adversarial_items()

        logger.info(
            "Evaluation run started: total=%s answerable=%s adversarial=%s k=%s gen_eval=%s",
            len(dataset), len(answerable), len(adversarial),
            self._config.k, self._config.run_generation_eval,
        )

        # --- 1. Retrieval evaluation ----------------------------------------
        retrieval_metrics = await self._run_retrieval(answerable)

        # --- 2. Generation evaluation (optional) ----------------------------
        generation_metrics: list[GenerationMetrics | None] = [None] * len(answerable)
        if self._config.run_generation_eval:
            generation_metrics = await self._run_generation(answerable, retrieval_metrics)

        # --- 3. Adversarial rejection rate ----------------------------------
        adv_rejection_rate = self._compute_adversarial_rate(adversarial)

        # --- 4. Assemble per-item results -----------------------------------
        item_results = self._build_item_results(
            answerable, retrieval_metrics, generation_metrics
        )
        # Add adversarial items (retrieval only, no generation)
        adv_retrieval = await self._run_retrieval(adversarial) if adversarial else []
        for item, ret in zip(adversarial, adv_retrieval):
            item_results.append(
                ItemEvaluationResult(
                    golden_item=item,
                    retrieval=ret,
                    generation=None,
                    adversarial_correctly_rejected=True,  # simulated — hook real logic here
                )
            )

        # --- 5. Aggregate ---------------------------------------------------
        overall = self._aggregate(
            [r for r in item_results if r.golden_item.category != QueryCategory.ADVERSARIAL.value]
        )
        per_category = self._per_category_reports(item_results)
        decision = self._make_decision(overall)

        logger.info(
            "Evaluation run complete: verdict=%s P@%s=%.3f R@%s=%.3f MRR=%.3f",
            decision.verdict,
            self._config.k, overall.precision_at_k,
            self._config.k, overall.recall_at_k,
            overall.mrr,
        )

        return EvaluationReport(
            overall=overall,
            per_category=per_category,
            adversarial_rejection_rate=adv_rejection_rate if adversarial else None,
            item_results=item_results,
            decision=decision,
            k=self._config.k,
        )

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    async def _run_retrieval(self, items: list[GoldenItem]) -> list[RetrievalMetrics]:
        if not items:
            return []
        try:
            return await self._retrieval_evaluator.evaluate(items)
        except RetrievalEvaluationError as exc:
            raise EvaluationServiceError(f"Retrieval evaluation failed: {exc}") from exc

    async def _run_generation(
        self,
        items: list[GoldenItem],
        retrieval_metrics: list[RetrievalMetrics],
    ) -> list[GenerationMetrics | None]:
        """Generate answers then evaluate them in a single pass."""
        assert self._generation_fn is not None

        async def _generate_and_score(
            item: GoldenItem, ret: RetrievalMetrics
        ) -> GenerationMetrics | None:
            # Fetch context by calling the generation function
            context = await self._fetch_context(ret.retrieved_ids)
            try:
                answer = await self._generation_fn(item.question, context)  # type: ignore[misc]
            except Exception as exc:
                logger.error(
                    "Generation failed for question %r: %s — skipping generation eval.",
                    item.question[:60], exc,
                )
                return None

            try:
                return await self._generation_evaluator.evaluate_one(
                    question=item.question,
                    context=context,
                    answer=answer,
                    ground_truth=item.ground_truth,
                )
            except GenerationEvaluationError as exc:
                logger.error(
                    "Generation evaluation failed for question %r: %s",
                    item.question[:60], exc,
                )
                return None

        results = await asyncio.gather(
            *[_generate_and_score(item, ret) for item, ret in zip(items, retrieval_metrics)]
        )
        return list(results)

    async def _fetch_context(self, doc_ids: list[str]) -> str:
        """
        Placeholder: join retrieved document IDs as context string.

        In a real deployment override this to fetch actual document content
        from your vector store using the retrieved IDs.
        """
        return " | ".join(doc_ids)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_item_results(
        items: list[GoldenItem],
        retrieval_metrics: list[RetrievalMetrics],
        generation_metrics: list[GenerationMetrics | None],
    ) -> list[ItemEvaluationResult]:
        return [
            ItemEvaluationResult(
                golden_item=item,
                retrieval=ret,
                generation=gen,
            )
            for item, ret, gen in zip(items, retrieval_metrics, generation_metrics)
        ]

    @staticmethod
    def _aggregate(results: list[ItemEvaluationResult]) -> AggregateMetrics:
        if not results:
            return AggregateMetrics(
                precision_at_k=0.0, recall_at_k=0.0, mrr=0.0, item_count=0
            )

        p_scores = [r.retrieval.precision_at_k for r in results]
        r_scores = [r.retrieval.recall_at_k for r in results]
        m_scores = [r.retrieval.mrr for r in results]
        ret_agg = aggregate_retrieval_scores(p_scores, r_scores, m_scores)

        gen_results = [r.generation for r in results if r.generation is not None]
        if gen_results:
            gen_agg = aggregate_generation_scores(
                [g.faithfulness for g in gen_results],
                [g.answer_relevancy for g in gen_results],
                [g.context_precision for g in gen_results],
                [g.context_recall for g in gen_results],
            )
        else:
            gen_agg = {}

        return AggregateMetrics(
            precision_at_k=ret_agg["precision_at_k"],
            recall_at_k=ret_agg["recall_at_k"],
            mrr=ret_agg["mrr"],
            faithfulness=gen_agg.get("faithfulness"),
            answer_relevancy=gen_agg.get("answer_relevancy"),
            context_precision=gen_agg.get("context_precision"),
            context_recall=gen_agg.get("context_recall"),
            item_count=len(results),
        )

    def _per_category_reports(
        self, item_results: list[ItemEvaluationResult]
    ) -> list[CategoryReport]:
        category_groups: dict[str, list[ItemEvaluationResult]] = {}
        for result in item_results:
            cat = result.golden_item.category
            category_groups.setdefault(cat, []).append(result)

        reports = []
        for category, items in sorted(category_groups.items()):
            metrics = self._aggregate(items)
            passed = self._check_thresholds(metrics)
            reports.append(
                CategoryReport(category=category, metrics=metrics, passed_thresholds=passed)
            )
        return reports

    def _check_thresholds(self, metrics: AggregateMetrics) -> bool:
        """Return True if all applicable metric scores meet their thresholds."""
        checks = {
            "precision_at_k": metrics.precision_at_k,
            "recall_at_k": metrics.recall_at_k,
            "mrr": metrics.mrr,
        }
        if metrics.faithfulness is not None:
            checks["faithfulness"] = metrics.faithfulness
        if metrics.answer_relevancy is not None:
            checks["answer_relevancy"] = metrics.answer_relevancy
        if metrics.context_precision is not None:
            checks["context_precision"] = metrics.context_precision
        if metrics.context_recall is not None:
            checks["context_recall"] = metrics.context_recall

        for metric, score in checks.items():
            threshold = self._config.thresholds.get(metric, 0.0)
            if score < threshold:
                return False
        return True

    def _make_decision(self, overall: AggregateMetrics) -> PassFailDecision:
        """Produce a pass/fail verdict against configured thresholds."""
        failing: list[str] = []

        metric_map = {
            "precision_at_k": overall.precision_at_k,
            "recall_at_k": overall.recall_at_k,
            "mrr": overall.mrr,
        }
        if overall.faithfulness is not None:
            metric_map["faithfulness"] = overall.faithfulness
        if overall.answer_relevancy is not None:
            metric_map["answer_relevancy"] = overall.answer_relevancy
        if overall.context_precision is not None:
            metric_map["context_precision"] = overall.context_precision
        if overall.context_recall is not None:
            metric_map["context_recall"] = overall.context_recall

        for metric, score in metric_map.items():
            threshold = self._config.thresholds.get(metric, 0.0)
            if score < threshold:
                failing.append(f"{metric}={score:.3f} < {threshold:.2f}")

        passed = len(failing) == 0
        return PassFailDecision(
            passed=passed,
            verdict="READY_TO_SHIP" if passed else "DO_NOT_SHIP",
            failing_metrics=failing,
        )

    @staticmethod
    def _compute_adversarial_rate(adversarial_items: list[GoldenItem]) -> float:
        """
        Compute the adversarial rejection rate.

        Currently returns 1.0 (all rejected) as a placeholder.
        Hook this into your intent classifier's output for real results.
        """
        if not adversarial_items:
            return 0.0
        # TODO: call query intent classifier and check answerable=False
        return 1.0
