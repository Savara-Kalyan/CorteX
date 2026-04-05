"""
Golden dataset management for RAG evaluation.

Responsibilities:
  GoldenDataset    — load, save, filter, and iterate golden items
  DatasetGenerator — LLM-assisted golden item generation from raw documents

Usage::

    # Load from a JSONL file
    dataset = GoldenDataset.from_jsonl("golden.jsonl")

    # Filter to a specific category
    factual = dataset.filter_by_category(QueryCategory.FACTUAL)

    # Generate new items with an LLM
    generator = DatasetGenerator()
    new_items = await generator.generate_from_documents(docs, n=10)
    dataset.extend(new_items)
    dataset.save_jsonl("golden.jsonl")
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

from rag.evaluation.models import Document, GoldenItem, QueryCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DatasetError(Exception):
    """Base exception for golden dataset operations."""


class DatasetLoadError(DatasetError):
    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"[Dataset] Failed to load '{path}': {reason}")


class DatasetSaveError(DatasetError):
    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"[Dataset] Failed to save '{path}': {reason}")


class DatasetGenerationError(DatasetError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"[Dataset] LLM generation failed: {reason}")


# ---------------------------------------------------------------------------
# GoldenDataset
# ---------------------------------------------------------------------------


class GoldenDataset:
    """
    An in-memory collection of GoldenItems with load/save/filter helpers.

    Args:
        items: Initial list of GoldenItem objects.
    """

    def __init__(self, items: list[GoldenItem] | None = None) -> None:
        self._items: list[GoldenItem] = items or []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "GoldenDataset":
        """
        Load golden items from a JSONL file (one JSON object per line).

        Raises:
            DatasetLoadError: on file not found or JSON parse errors.
        """
        path = Path(path)
        if not path.exists():
            raise DatasetLoadError(str(path), "file not found")

        items: list[GoldenItem] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(GoldenItem.model_validate_json(line))
                    except Exception as exc:
                        logger.warning(
                            "Skipping malformed line %s in %s: %s", line_no, path, exc
                        )
        except OSError as exc:
            raise DatasetLoadError(str(path), str(exc)) from exc

        logger.info("Loaded %s golden items from %s", len(items), path)
        return cls(items)

    @classmethod
    def from_list(cls, items: list[GoldenItem]) -> "GoldenDataset":
        """Construct directly from a list of GoldenItem objects."""
        return cls(list(items))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_jsonl(self, path: str | Path) -> None:
        """
        Persist all items to a JSONL file.

        Raises:
            DatasetSaveError: on write failure.
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                for item in self._items:
                    fh.write(item.model_dump_json() + "\n")
        except OSError as exc:
            raise DatasetSaveError(str(path), str(exc)) from exc

        logger.info("Saved %s golden items to %s", len(self._items), path)

    # ------------------------------------------------------------------
    # Access / mutation
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index: int) -> GoldenItem:
        return self._items[index]

    def extend(self, items: list[GoldenItem]) -> None:
        """Append items to the dataset."""
        self._items.extend(items)
        logger.debug("Extended dataset: total_items=%s", len(self._items))

    def filter_by_category(self, category: QueryCategory) -> list[GoldenItem]:
        """Return items matching the given category."""
        return [i for i in self._items if i.category == category.value]

    def answerable_items(self) -> list[GoldenItem]:
        """Return all non-adversarial items (have ground-truth answers)."""
        return [i for i in self._items if i.category != QueryCategory.ADVERSARIAL.value]

    def adversarial_items(self) -> list[GoldenItem]:
        """Return only adversarial (unanswerable) items."""
        return [i for i in self._items if i.category == QueryCategory.ADVERSARIAL.value]

    def category_counts(self) -> dict[str, int]:
        """Return a mapping of category → item count."""
        counts: dict[str, int] = {}
        for item in self._items:
            counts[item.category] = counts.get(item.category, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# DatasetGenerator — LLM-assisted golden item generation
# ---------------------------------------------------------------------------


class DatasetGenerator:
    """
    Generate golden evaluation items from raw documents using an LLM.

    The generator calls the configured LLM to:
      1. Create a realistic question a user would ask about the document.
      2. Answer that question using only the document text.
      3. Classify the question into a QueryCategory.

    Args:
        llm_caller: Async callable ``(prompt: str) -> str``.  When None the
                    generator falls back to ``settings`` to build an OpenAI
                    caller automatically.
    """

    _QUESTION_PROMPT = """You are building a RAG evaluation dataset.

Document title: {title}
Document content:
{content}

Generate ONE realistic question that a business user would ask, whose answer
can be found entirely within the document above.

Respond with a JSON object on a single line with these exact keys:
  question   — the question string
  answer     — the answer using ONLY the document text
  category   — one of: factual, procedural, comparative, analytical

Example:
{{"question": "What is the refund policy?", "answer": "Refunds are processed within 5 business days.", "category": "procedural"}}

JSON:"""

    def __init__(self, llm_caller: Callable[[str], any] | None = None) -> None:
        self._call_llm = llm_caller or self._build_default_caller()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_from_documents(
        self,
        documents: list[Document],
        n: int = 10,
        adversarial_ratio: float = 0.2,
        adversarial_questions: list[str] | None = None,
    ) -> list[GoldenItem]:
        """
        Generate ``n`` golden items from the provided documents.

        ``adversarial_ratio`` controls what fraction of the final set are
        adversarial (out-of-scope) questions that should be rejected.

        Args:
            documents:             Source corpus documents.
            n:                     Total items to generate (answerable + adversarial).
            adversarial_ratio:     Fraction reserved for adversarial items (default 0.2).
            adversarial_questions: Optional explicit list of adversarial questions.
                                   If not supplied, generic placeholders are used.

        Returns:
            List of GoldenItem objects.

        Raises:
            DatasetGenerationError: when LLM calls fail for all documents.
        """
        if not documents:
            raise DatasetGenerationError("No documents provided for generation.")

        n_adversarial = max(1, round(n * adversarial_ratio))
        n_answerable = n - n_adversarial

        answerable_items = await self._generate_answerable(documents, n_answerable)
        adversarial_items = self._build_adversarial(n_adversarial, adversarial_questions)

        total = answerable_items + adversarial_items
        logger.info(
            "Dataset generation complete: answerable=%s adversarial=%s total=%s",
            len(answerable_items), len(adversarial_items), len(total),
        )
        return total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_answerable(
        self, documents: list[Document], n: int
    ) -> list[GoldenItem]:
        import random

        items: list[GoldenItem] = []
        pool = list(documents) * ((n // len(documents)) + 2)
        random.shuffle(pool)

        for doc in pool:
            if len(items) >= n:
                break
            item = await self._generate_one(doc)
            if item is not None:
                items.append(item)

        if not items:
            raise DatasetGenerationError(
                "LLM failed to generate any answerable items from the provided documents."
            )
        return items[:n]

    async def _generate_one(self, doc: Document) -> GoldenItem | None:
        prompt = self._QUESTION_PROMPT.format(title=doc.title, content=doc.content)
        try:
            raw = await self._call_llm(prompt)
            parsed = self._parse_llm_json(raw)
            if parsed is None:
                return None

            category_str = parsed.get("category", "factual").lower()
            try:
                category = QueryCategory(category_str)
            except ValueError:
                category = QueryCategory.FACTUAL

            return GoldenItem(
                question=parsed["question"],
                ground_truth=parsed["answer"],
                relevant_doc_ids=[doc.id],
                category=category,
            )
        except DatasetGenerationError:
            raise
        except Exception as exc:
            logger.warning("Failed to generate item for doc '%s': %s", doc.id, exc)
            return None

    @staticmethod
    def _parse_llm_json(raw: str) -> dict | None:
        """Extract a JSON object from raw LLM output."""
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Failed to parse LLM JSON response: %s | raw=%r", exc, raw[:200])
        return None

    @staticmethod
    def _build_adversarial(n: int, questions: list[str] | None) -> list[GoldenItem]:
        defaults = [
            "What is the current stock price?",
            "What is the weather forecast for tomorrow?",
            "Tell me about our competitor's pricing strategy.",
            "What will happen to the economy next year?",
            "Can you write me a poem?",
        ]
        source = questions or defaults
        pool = (source * ((n // len(source)) + 2))[:n]
        return [
            GoldenItem(
                question=q,
                ground_truth="UNANSWERABLE — not in knowledge base",
                relevant_doc_ids=[],
                category=QueryCategory.ADVERSARIAL,
            )
            for q in pool
        ]

    @staticmethod
    def _build_default_caller():
        """Build an async LLM caller backed by OpenAI (from project settings)."""
        from settings import settings

        async def _call(prompt: str) -> str:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI()
                resp = await client.chat.completions.create(
                    model=settings.llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                raise DatasetGenerationError(f"OpenAI call failed: {exc}") from exc

        return _call
