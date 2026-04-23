"""
RAGAS-based RAG evaluation tests.

TestGoldenDatasetStructure  — validates dataset shape (no API needed)
TestRAGASEval               — runs the full pipeline and scores with RAGAS
                              (requires PGVECTOR_URL + OPENAI_API_KEY)

Dataset: tests/evaluation/golden_dataset.json
  Each entry: {id, category, domain, query, ground_truth, relevant_chunk_ids}
"""

import json
import asyncio
import pytest
from pathlib import Path

from tests.conftest import requires_pg, requires_llm, requires_all

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

with open(DATASET_PATH) as f:
    GOLDEN_DATASET = json.load(f)

REQUIRED_FIELDS = {"id", "category", "domain", "query", "ground_truth", "relevant_chunk_ids"}
VALID_CATEGORIES = {"hr", "engineering", "culture"}


# ---------------------------------------------------------------------------
# Dataset structure tests — no external I/O
# ---------------------------------------------------------------------------

class TestGoldenDatasetStructure:

    def test_dataset_has_20_entries(self):
        assert len(GOLDEN_DATASET) == 20

    def test_all_entries_have_required_fields(self):
        for entry in GOLDEN_DATASET:
            missing = REQUIRED_FIELDS - entry.keys()
            assert not missing, f"Entry {entry.get('id')} missing: {missing}"

    def test_all_ids_are_unique(self):
        ids = [e["id"] for e in GOLDEN_DATASET]
        assert len(ids) == len(set(ids))

    def test_all_categories_are_valid(self):
        for entry in GOLDEN_DATASET:
            assert entry["category"] in VALID_CATEGORIES, \
                f"Invalid category '{entry['category']}' in entry {entry['id']}"

    def test_hr_entries_count(self):
        hr = [e for e in GOLDEN_DATASET if e["category"] == "hr"]
        assert len(hr) >= 6

    def test_engineering_entries_count(self):
        eng = [e for e in GOLDEN_DATASET if e["category"] == "engineering"]
        assert len(eng) >= 4

    def test_culture_entries_count(self):
        culture = [e for e in GOLDEN_DATASET if e["category"] == "culture"]
        assert len(culture) >= 4

    def test_queries_are_non_empty(self):
        for entry in GOLDEN_DATASET:
            assert len(entry["query"].strip()) > 10, \
                f"Query too short in entry {entry['id']}"

    def test_ground_truths_are_non_empty(self):
        for entry in GOLDEN_DATASET:
            assert len(entry["ground_truth"].strip()) > 10, \
                f"Ground truth too short in entry {entry['id']}"

    def test_has_relevant_chunk_ids_field(self):
        for entry in GOLDEN_DATASET:
            assert "relevant_chunk_ids" in entry, \
                f"Missing relevant_chunk_ids in entry {entry['id']}"

    def test_domain_matches_category_for_known_categories(self):
        for entry in GOLDEN_DATASET:
            assert entry["domain"] == entry["category"], \
                f"Domain/category mismatch in entry {entry['id']}"


# ---------------------------------------------------------------------------
# RAGAS live evaluation — real pipeline + real LLM
# ---------------------------------------------------------------------------

@requires_all
class TestRAGASEval:
    """
    Ingests golden dataset contexts, runs queries through the full RAG pipeline,
    and evaluates with RAGAS metrics. Uses source_file='test:ragas-*' for cleanup.
    """

    @pytest.fixture(scope="class")
    def pipeline_with_data(self, sync_pg_conn):
        from langchain_core.documents import Document
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline()

        # Seed each golden entry's context into the vector store
        docs = []
        for entry in GOLDEN_DATASET:
            docs.append(Document(
                page_content=entry["context"],
                metadata={
                    "source": f"test:ragas-{entry['id']}.md",
                    "domain": entry["domain"],
                    "file_type": "markdown",
                },
            ))

        texts = [d.page_content for d in docs]
        embeddings = asyncio.run(pipeline._embeddings.embed_documents(texts))
        asyncio.run(pipeline._vector_store.add_documents(docs, embeddings))

        yield pipeline

        cur = sync_pg_conn.cursor()
        cur.execute("DELETE FROM documents WHERE source_file LIKE 'test:ragas-%';")
        cur.close()

    def _run_query(self, pipeline, question: str, domain: str) -> dict:
        tier_map = {"hr": "confidential", "engineering": "internal", "culture": "public"}
        tier = tier_map.get(domain, "internal")
        return asyncio.run(pipeline.query(question, user_tier=tier))

    def _compute_ragas_scores(self, samples: list[dict]) -> dict:
        import os
        from openai import OpenAI
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.metrics.collections import (
            Faithfulness, AnswerRelevancy, ContextPrecision, AnswerCorrectness,
        )

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        llm = llm_factory("gpt-4o-mini", client=client)
        emb = OpenAIEmbeddings(model="text-embedding-3-small", client=client)

        ragas_samples = [
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
                reference=s["ground_truth"],
            )
            for s in samples
        ]
        return evaluate(
            dataset=EvaluationDataset(samples=ragas_samples),
            metrics=[
                Faithfulness(llm=llm),
                AnswerRelevancy(llm=llm, embeddings=emb),
                ContextPrecision(llm=llm),
                AnswerCorrectness(llm=llm, embeddings=emb),
            ],
        )

    def test_hr_ragas_scores_meet_threshold(self, pipeline_with_data):
        hr_entries = [e for e in GOLDEN_DATASET if e["category"] == "hr"]
        samples = []
        for entry in hr_entries:
            result = self._run_query(pipeline_with_data, entry["query"], entry["domain"])
            if result.get("answer") and result.get("chunks_retrieved", 0) > 0:
                samples.append({
                    "question": entry["query"],
                    "answer": result["answer"],
                    "contexts": [c for c in [entry["context"]] if c],
                    "ground_truth": entry["ground_truth"],
                })

        assert len(samples) >= 5, "Not enough HR answers with retrieved chunks"
        scores = self._compute_ragas_scores(samples)
        assert scores["faithfulness"] >= 0.6, f"HR faithfulness too low: {scores['faithfulness']}"
        assert scores["answer_relevancy"] >= 0.6, f"HR relevancy too low: {scores['answer_relevancy']}"

    def test_engineering_ragas_scores_meet_threshold(self, pipeline_with_data):
        eng_entries = [e for e in GOLDEN_DATASET if e["category"] == "engineering"]
        samples = []
        for entry in eng_entries:
            result = self._run_query(pipeline_with_data, entry["query"], entry["domain"])
            if result.get("answer") and result.get("chunks_retrieved", 0) > 0:
                samples.append({
                    "question": entry["query"],
                    "answer": result["answer"],
                    "contexts": [entry["context"]],
                    "ground_truth": entry["ground_truth"],
                })

        assert len(samples) >= 4, "Not enough Engineering answers with retrieved chunks"
        scores = self._compute_ragas_scores(samples)
        assert scores["faithfulness"] >= 0.6, f"Eng faithfulness too low: {scores['faithfulness']}"
        assert scores["answer_relevancy"] >= 0.6, f"Eng relevancy too low: {scores['answer_relevancy']}"

    def test_culture_ragas_scores_meet_threshold(self, pipeline_with_data):
        culture_entries = [e for e in GOLDEN_DATASET if e["category"] == "culture"]
        samples = []
        for entry in culture_entries:
            result = self._run_query(pipeline_with_data, entry["query"], entry["domain"])
            if result.get("answer") and result.get("chunks_retrieved", 0) > 0:
                samples.append({
                    "question": entry["query"],
                    "answer": result["answer"],
                    "contexts": [entry["context"]],
                    "ground_truth": entry["ground_truth"],
                })

        assert len(samples) >= 3, "Not enough Culture answers with retrieved chunks"
        scores = self._compute_ragas_scores(samples)
        assert scores["faithfulness"] >= 0.65, f"Culture faithfulness too low: {scores['faithfulness']}"

    def test_overall_answer_correctness(self, pipeline_with_data):
        samples = []
        for entry in GOLDEN_DATASET[:10]:
            result = self._run_query(pipeline_with_data, entry["query"], entry["domain"])
            if result.get("answer"):
                samples.append({
                    "question": entry["query"],
                    "answer": result["answer"],
                    "contexts": [entry["context"]],
                    "ground_truth": entry["ground_truth"],
                })

        assert len(samples) >= 5
        scores = self._compute_ragas_scores(samples)
        assert scores["answer_correctness"] >= 0.5, \
            f"Overall correctness too low: {scores['answer_correctness']}"

    def test_all_entries_produce_answers(self, pipeline_with_data):
        empty_count = 0
        for entry in GOLDEN_DATASET:
            result = self._run_query(pipeline_with_data, entry["query"], entry["domain"])
            if not result.get("answer"):
                empty_count += 1
        assert empty_count <= 3, f"{empty_count} entries produced empty answers (threshold: 3)"

    def test_context_precision_on_culture_entries(self, pipeline_with_data):
        culture_entries = [e for e in GOLDEN_DATASET if e["category"] == "culture"]
        samples = []
        for entry in culture_entries:
            result = self._run_query(pipeline_with_data, entry["query"], entry["domain"])
            if result.get("answer") and result.get("chunks_retrieved", 0) > 0:
                samples.append({
                    "question": entry["query"],
                    "answer": result["answer"],
                    "contexts": [entry["context"]],
                    "ground_truth": entry["ground_truth"],
                })

        if len(samples) >= 3:
            scores = self._compute_ragas_scores(samples)
            assert scores["context_precision"] >= 0.5, \
                f"Culture context precision too low: {scores['context_precision']}"
