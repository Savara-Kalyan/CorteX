"""
Run evaluation against the golden dataset and update README.md with results.

Usage:
    python evaluate.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
load_dotenv()

from observability.logger import configure, get_logging_config
configure(get_logging_config())

from rag.pipeline import RAGPipeline
from rag.evaluation import evaluate_batch

DATASET_PATH = Path("tests/evaluation/golden_dataset.json")
README_PATH = Path("README.md")

TIER_MAP = {"hr": "confidential", "engineering": "internal", "culture": "public"}




async def collect_pipeline_results(pipeline: RAGPipeline, dataset: list[dict]) -> list[dict]:
    records = []
    for i, entry in enumerate(dataset, 1):
        tier = TIER_MAP.get(entry["domain"], "internal")
        result = await pipeline.query(entry["query"], user_tier=tier, domain=entry["domain"])
        records.append({**entry, "pipeline_result": result})
        print(f"  [{i:02d}/{len(dataset)}] {entry['id']}", flush=True)
    return records


def compute_retrieval_metrics(records: list[dict]) -> dict:
    results_by_cat: dict[str, list] = {}
    for rec in records:
        result = rec["pipeline_result"]
        relevant_ids = set(rec.get("relevant_chunk_ids", []))
        retrieved_ids = result.get("chunk_ids", [])
        results_by_cat.setdefault(rec["category"], []).append({
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
        })

    overall, per_cat = [], {}
    for cat, items in results_by_cat.items():
        per_cat[cat] = evaluate_batch(items, k=5)
        overall.extend(items)

    return {"overall": evaluate_batch(overall, k=5), "per_category": per_cat}


def compute_ragas_metrics(records: list[dict]) -> dict:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._answer_correctness import AnswerCorrectness

        llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

        faithfulness = Faithfulness(llm=llm)
        answer_relevancy = AnswerRelevancy(llm=llm, embeddings=emb)
        context_precision = ContextPrecision(llm=llm)
        answer_correctness = AnswerCorrectness(llm=llm, embeddings=emb)

        samples = [
            {
                "question": rec["query"],
                "answer": rec["pipeline_result"].get("answer", ""),
                "contexts": rec["pipeline_result"].get("contexts") or [rec["ground_truth"]],
                "ground_truth": rec["ground_truth"],
            }
            for rec in records
            if rec["pipeline_result"].get("answer")
        ]

        from ragas.run_config import RunConfig
        result = evaluate(
            Dataset.from_list(samples),
            metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
            run_config=RunConfig(timeout=180, max_retries=3, max_workers=4),
        )
    df = result.to_pandas()
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]
    return {col: round(float(df[col].mean()), 4) for col in metric_cols if col in df.columns}


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def update_readme(retrieval: dict, ragas: dict, timestamp: str) -> None:
    ov = retrieval["overall"]
    p5   = ov.get("P@5",    0.0)
    mrr  = ov.get("MRR",    0.0)
    ndcg = ov.get("NDCG@5", 0.0)

    block = f"""\
<!-- eval-results-start -->
## Measured Results
*Last evaluated: {timestamp} — golden dataset n=20*

### Retrieval Quality (word-overlap scoring)

| Metric | Target | Measured |
|---|---|---|
| **P@5** | >= 0.25 | **{_fmt(p5)}** |
| **MRR** | >= 0.50 | **{_fmt(mrr)}** |
| **NDCG@5** | >= 0.50 | **{_fmt(ndcg)}** |

### RAGAS Scores (LLM-as-judge)

| Metric | Target | Score |
|---|---|---|
| **Faithfulness** | >= 0.60 | **{_fmt(ragas.get('faithfulness', 0))}** |
| **Answer Relevancy** | >= 0.60 | **{_fmt(ragas.get('answer_relevancy', 0))}** |
| **Context Precision** | >= 0.50 | **{_fmt(ragas.get('context_precision', 0))}** |
| **Answer Correctness** | >= 0.50 | **{_fmt(ragas.get('answer_correctness', 0))}** |
<!-- eval-results-end -->"""

    text = README_PATH.read_text(encoding="utf-8")
    marker_start = "<!-- eval-results-start -->"
    marker_end = "<!-- eval-results-end -->"

    if marker_start in text:
        text = re.sub(
            re.escape(marker_start) + r".*?" + re.escape(marker_end),
            block,
            text,
            flags=re.DOTALL,
        )
    else:
        text += f"\n\n---\n\n{block}\n"

    README_PATH.write_text(text, encoding="utf-8")
    print(f"\nREADME.md updated  [{timestamp}]")


async def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text())
    print(f"Loaded {len(dataset)} entries from golden dataset\n")

    pipeline = RAGPipeline()

    print("Running pipeline queries...")
    records = await collect_pipeline_results(pipeline, dataset)

    print("\nComputing retrieval metrics (word-overlap)...")
    retrieval = compute_retrieval_metrics(records)
    ov = retrieval["overall"]
    print(f"  Overall  P@5={ov['P@5']:.4f}  MRR={ov['MRR']:.4f}  NDCG@5={ov['NDCG@5']:.4f}")
    for cat, s in sorted(retrieval["per_category"].items()):
        print(f"  {cat:<12} P@5={s['P@5']:.4f}  MRR={s['MRR']:.4f}  NDCG@5={s['NDCG@5']:.4f}")

    print("\nComputing RAGAS metrics (LLM-as-judge)...")
    ragas = compute_ragas_metrics(records)
    for k, v in ragas.items():
        print(f"  {k}: {v:.4f}")

    IST = timezone(timedelta(hours=5, minutes=30))
    timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    update_readme(retrieval, ragas, timestamp)


if __name__ == "__main__":
    asyncio.run(main())
