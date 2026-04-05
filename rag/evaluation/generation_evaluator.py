"""
Generation evaluation module (RAGAS-style).

Evaluates four generation quality dimensions via LLM-as-a-judge:
  1. Faithfulness       — are all answer claims grounded in the retrieved context?
  2. Answer Relevancy   — does the answer actually address the question?
  3. Context Precision  — are the retrieved chunks relevant to the question?
  4. Context Recall     — does the context contain all information needed?

Each dimension is scored 0.0–1.0 by the LLM.  Scores below 0.0 or above 1.0
returned by the model are clamped. Parse failures produce a score of 0.0 and
are logged at WARNING level.

Usage::

    evaluator = GenerationEvaluator()
    metrics = await evaluator.evaluate_one(
        question="What is the refund policy?",
        context="Refunds are processed within 5 business days...",
        answer="Refunds take 5 business days to process.",
        ground_truth="Refunds are processed within 5 business days.",
    )
    print(metrics.faithfulness, metrics.answer_relevancy)
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Callable, Awaitable

from rag.evaluation.models import GenerationMetrics

logger = logging.getLogger(__name__)

# Type alias
LLMCaller = Callable[[str], Awaitable[str]]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GenerationEvaluationError(Exception):
    """Raised when generation evaluation cannot proceed."""


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are a RAG evaluation judge.

Context (retrieved documents):
{context}

Answer to evaluate:
{answer}

Task: Score FAITHFULNESS — what fraction of the answer's factual claims are
supported by the context above?
A claim is supported if the context contains information that confirms it.
Ignore claims that are generic or not verifiable.

Reply with a single decimal number between 0.0 and 1.0.
Examples: 0.0  0.5  0.85  1.0

Score:"""

_ANSWER_RELEVANCY_PROMPT = """\
You are a RAG evaluation judge.

Question: {question}

Answer to evaluate:
{answer}

Task: Score ANSWER RELEVANCY — how well does the answer address the question?
1.0 = fully addresses the question with precise, complete information.
0.0 = entirely off-topic or refuses to answer without reason.

Reply with a single decimal number between 0.0 and 1.0.

Score:"""

_CONTEXT_PRECISION_PROMPT = """\
You are a RAG evaluation judge.

Question: {question}

Retrieved context:
{context}

Task: Score CONTEXT PRECISION — what fraction of the retrieved context is
actually useful and relevant for answering the question?
Penalise noisy, off-topic, or redundant chunks.

Reply with a single decimal number between 0.0 and 1.0.

Score:"""

_CONTEXT_RECALL_PROMPT = """\
You are a RAG evaluation judge.

Question: {question}

Ground-truth answer: {ground_truth}

Retrieved context:
{context}

Task: Score CONTEXT RECALL — does the retrieved context contain ALL the
information required to produce the ground-truth answer?
1.0 = all needed information is present.
0.0 = the context is missing critical information.

Reply with a single decimal number between 0.0 and 1.0.

Score:"""


# ---------------------------------------------------------------------------
# GenerationEvaluator
# ---------------------------------------------------------------------------


class GenerationEvaluator:
    """
    LLM-as-a-judge evaluator for RAG generation quality.

    Args:
        llm_caller:  Async callable ``(prompt: str) -> str``.
                     When None the evaluator auto-builds an OpenAI caller
                     from project settings.
        concurrency: Maximum parallel LLM calls per ``evaluate_batch`` call
                     (default 4).
    """

    def __init__(
        self,
        llm_caller: LLMCaller | None = None,
        concurrency: int = 4,
    ) -> None:
        self._call_llm = llm_caller or self._build_default_caller()
        self._semaphore = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate_one(
        self,
        question: str,
        context: str,
        answer: str,
        ground_truth: str,
    ) -> GenerationMetrics:
        """
        Evaluate generation quality for a single (question, context, answer) triple.

        All four metrics are scored concurrently.  Individual LLM call failures
        produce a score of 0.0 for that metric and are logged.

        Args:
            question:     The original user question.
            context:      Retrieved document text joined as a single string.
            answer:       The RAG system's generated answer.
            ground_truth: The expected ground-truth answer.

        Returns:
            GenerationMetrics with scores in [0, 1] for all four dimensions.
        """
        faithfulness, relevancy, precision, recall = await asyncio.gather(
            self._score_faithfulness(answer, context),
            self._score_answer_relevancy(question, answer),
            self._score_context_precision(question, context),
            self._score_context_recall(question, context, ground_truth),
        )

        logger.debug(
            "Generation metrics: question=%r faith=%.3f relev=%.3f prec=%.3f recall=%.3f",
            question[:60], faithfulness, relevancy, precision, recall,
        )

        return GenerationMetrics(
            question=question,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
        )

    async def evaluate_batch(
        self,
        items: list[dict],
    ) -> list[GenerationMetrics]:
        """
        Evaluate a batch of generation samples.

        Each dict must have keys: ``question``, ``context``, ``answer``,
        ``ground_truth``.

        Failures for individual items are caught and produce zero-score metrics
        so the overall batch still completes.

        Args:
            items: List of dicts with the required keys.

        Returns:
            List of GenerationMetrics in the same order as ``items``.

        Raises:
            GenerationEvaluationError: when ``items`` is empty.
        """
        if not items:
            raise GenerationEvaluationError("No items provided for generation evaluation.")

        logger.info("Starting generation evaluation batch: items=%s", len(items))

        async def _safe(item: dict) -> GenerationMetrics:
            try:
                return await self.evaluate_one(
                    question=item["question"],
                    context=item["context"],
                    answer=item["answer"],
                    ground_truth=item["ground_truth"],
                )
            except Exception as exc:
                logger.error(
                    "Generation evaluation failed for question %r: %s — zero scores assigned.",
                    item.get("question", "")[:60], exc,
                )
                return GenerationMetrics(
                    question=item.get("question", ""),
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                )

        results = await asyncio.gather(*[_safe(i) for i in items])
        logger.info("Generation evaluation batch complete: evaluated=%s", len(results))
        return list(results)

    # ------------------------------------------------------------------
    # Metric scorers
    # ------------------------------------------------------------------

    async def _score_faithfulness(self, answer: str, context: str) -> float:
        prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        return await self._llm_score(prompt, "faithfulness")

    async def _score_answer_relevancy(self, question: str, answer: str) -> float:
        prompt = _ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)
        return await self._llm_score(prompt, "answer_relevancy")

    async def _score_context_precision(self, question: str, context: str) -> float:
        prompt = _CONTEXT_PRECISION_PROMPT.format(question=question, context=context)
        return await self._llm_score(prompt, "context_precision")

    async def _score_context_recall(
        self, question: str, context: str, ground_truth: str
    ) -> float:
        prompt = _CONTEXT_RECALL_PROMPT.format(
            question=question, context=context, ground_truth=ground_truth
        )
        return await self._llm_score(prompt, "context_recall")

    # ------------------------------------------------------------------
    # LLM call + score parsing
    # ------------------------------------------------------------------

    async def _llm_score(self, prompt: str, metric_name: str) -> float:
        """Call the LLM, parse a float score from its response."""
        async with self._semaphore:
            try:
                raw = await self._call_llm(prompt)
            except Exception as exc:
                logger.warning("LLM call failed for metric '%s': %s", metric_name, exc)
                return 0.0

        score = self._parse_score(raw)
        if score is None:
            logger.warning(
                "Could not parse score for metric '%s' from response: %r",
                metric_name, raw[:100],
            )
            return 0.0

        return max(0.0, min(1.0, score))  # clamp to [0, 1]

    @staticmethod
    def _parse_score(raw: str) -> float | None:
        """Extract the first decimal number from an LLM response string."""
        match = re.search(r"\b(1\.0+|0?\.\d+|\d+\.\d+)\b", raw.strip())
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        # Try bare integer 0 or 1
        match = re.search(r"\b([01])\b", raw.strip())
        if match:
            return float(match.group())
        return None

    # ------------------------------------------------------------------
    # Default LLM caller
    # ------------------------------------------------------------------

    @staticmethod
    def _build_default_caller() -> LLMCaller:
        """Build an async LLM caller backed by OpenAI from project settings."""
        from settings import settings

        async def _call(prompt: str) -> str:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI()
                resp = await client.chat.completions.create(
                    model=settings.llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                raise GenerationEvaluationError(
                    f"OpenAI call failed: {exc}"
                ) from exc

        return _call
