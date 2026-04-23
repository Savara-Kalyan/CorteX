"""
Prometheus metrics for CorteX.

Exposes an HTTP /metrics endpoint (default port 8001) that Prometheus scrapes.
Also keeps an in-process MetricsDashboard for print_summary() debugging.

Metrics exported:
  cortex_agent_calls_total          counter  agent
  cortex_query_duration_seconds     histogram
  cortex_tokens_used_total          counter  model, token_type
  cortex_cost_usd_total             counter  user_id
  cortex_chunks_retrieved           histogram
  cortex_rag_precision_at5          gauge
  cortex_rag_mrr                    gauge
  cortex_rag_ndcg                   gauge
  cortex_active_requests            gauge
  cortex_errors_total               counter  agent, error_type
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from reliability.cost_tracker import CostTracker
from observability.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metric definitions  (module-level singletons)
# ---------------------------------------------------------------------------

AGENT_CALLS = Counter(
    "cortex_agent_calls_total",
    "Total agent invocations",
    ["agent"],
)

QUERY_DURATION = Histogram(
    "cortex_query_duration_seconds",
    "End-to-end query latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

TOKENS_USED = Counter(
    "cortex_tokens_used_total",
    "LLM tokens consumed",
    ["model", "token_type"],  # token_type: input | output | embedding
)

COST_USD = Counter(
    "cortex_cost_usd_total",
    "USD cost incurred",
    ["user_id"],
)

CHUNKS_RETRIEVED = Histogram(
    "cortex_chunks_retrieved",
    "Number of RAG chunks retrieved per query",
    buckets=[0, 1, 2, 3, 5, 8, 13, 20],
)

RAG_PRECISION = Gauge("cortex_rag_precision_at5", "Rolling P@5 over recent queries")
RAG_MRR = Gauge("cortex_rag_mrr", "Rolling Mean Reciprocal Rank")
RAG_NDCG = Gauge("cortex_rag_ndcg", "Rolling NDCG")

ACTIVE_REQUESTS = Gauge("cortex_active_requests", "Queries currently in-flight")

ERRORS = Counter(
    "cortex_errors_total",
    "Errors by agent and type",
    ["agent", "error_type"],
)

# ---------------------------------------------------------------------------
# Legacy dataclasses (kept for print_summary)
# ---------------------------------------------------------------------------

@dataclass
class RetrievalMetrics:
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    total_queries: int = 0
    avg_chunks_retrieved: float = 0.0


@dataclass
class CostMetrics:
    user_id: str = "system"
    daily_total_usd: float = 0.0
    daily_budget_usd: float = 2.0
    budget_remaining_pct: float = 100.0


@dataclass
class Dashboard:
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    cost: CostMetrics = field(default_factory=CostMetrics)
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    agent_call_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MetricsDashboard
# ---------------------------------------------------------------------------

class MetricsDashboard:
    """
    Records metrics into Prometheus and maintains in-process buffers
    for the print_summary() debug display.

    Call start_metrics_server() once at startup to expose /metrics.
    """

    _server_started = False
    _server_lock = threading.Lock()

    def __init__(self, daily_budget: float = 2.0):
        self._cost_tracker = CostTracker()
        self._daily_budget = daily_budget
        self._agent_calls: dict[str, int] = {}
        self._retrieval_buffer: list[dict] = []

    # ------------------------------------------------------------------
    # Prometheus server
    # ------------------------------------------------------------------

    @classmethod
    def start_metrics_server(cls, port: int = 8001) -> None:
        """Start the Prometheus HTTP /metrics server (idempotent)."""
        with cls._server_lock:
            if cls._server_started:
                return
            start_http_server(port)
            cls._server_started = True
            logger.info(f"Prometheus metrics server started on :{port}/metrics")

    # ------------------------------------------------------------------
    # Recording — updates both Prometheus and in-process buffers
    # ------------------------------------------------------------------

    def record_agent_call(self, agent: str) -> None:
        AGENT_CALLS.labels(agent=agent).inc()
        self._agent_calls[agent] = self._agent_calls.get(agent, 0) + 1

    def record_query_duration(self, seconds: float) -> None:
        QUERY_DURATION.observe(seconds)

    def record_tokens(self, model: str, input_tokens: int, output_tokens: int = 0) -> None:
        if input_tokens:
            TOKENS_USED.labels(model=model, token_type="input").inc(input_tokens)
        if output_tokens:
            TOKENS_USED.labels(model=model, token_type="output").inc(output_tokens)

    def record_embedding_tokens(self, model: str, tokens: int) -> None:
        TOKENS_USED.labels(model=model, token_type="embedding").inc(tokens)

    def record_cost(self, user_id: str, amount_usd: float) -> None:
        if amount_usd > 0:
            COST_USD.labels(user_id=user_id).inc(amount_usd)

    def record_retrieval(
        self,
        relevant_retrieved: int,
        total_retrieved: int,
        reciprocal_rank: float,
        ndcg: float,
    ) -> None:
        CHUNKS_RETRIEVED.observe(total_retrieved)
        self._retrieval_buffer.append({
            "relevant": relevant_retrieved,
            "total": total_retrieved,
            "reciprocal_rank": reciprocal_rank,
            "ndcg": ndcg,
        })
        m = self._compute_retrieval_metrics()
        RAG_PRECISION.set(m.precision_at_5)
        RAG_MRR.set(m.mrr)
        RAG_NDCG.set(m.ndcg)

    def record_error(self, agent: str, error_type: str) -> None:
        ERRORS.labels(agent=agent, error_type=error_type).inc()

    def inc_active_requests(self) -> None:
        ACTIVE_REQUESTS.inc()

    def dec_active_requests(self) -> None:
        ACTIVE_REQUESTS.dec()

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    async def get_dashboard(self, user_id: str = "system") -> Dashboard:
        daily_total = await self._cost_tracker.get_daily_total(user_id)
        remaining_pct = max(0.0, (1 - daily_total / self._daily_budget) * 100)
        retrieval = self._compute_retrieval_metrics()
        return Dashboard(
            cost=CostMetrics(
                user_id=user_id,
                daily_total_usd=daily_total,
                daily_budget_usd=self._daily_budget,
                budget_remaining_pct=remaining_pct,
            ),
            retrieval=retrieval,
            agent_call_counts=dict(self._agent_calls),
        )

    def _compute_retrieval_metrics(self) -> RetrievalMetrics:
        if not self._retrieval_buffer:
            return RetrievalMetrics()
        n = len(self._retrieval_buffer)
        p5 = sum(r["relevant"] / max(r["total"], 1) for r in self._retrieval_buffer) / n
        mrr = sum(r["reciprocal_rank"] for r in self._retrieval_buffer) / n
        ndcg = sum(r["ndcg"] for r in self._retrieval_buffer) / n
        avg_chunks = sum(r["total"] for r in self._retrieval_buffer) / n
        return RetrievalMetrics(
            precision_at_5=round(p5, 4),
            mrr=round(mrr, 4),
            ndcg=round(ndcg, 4),
            total_queries=n,
            avg_chunks_retrieved=round(avg_chunks, 2),
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    async def print_summary(self, user_id: str = "system") -> None:
        d = await self.get_dashboard(user_id)
        print("\n" + "=" * 60)
        print(f"  CORTEX METRICS DASHBOARD  [{d.timestamp}]")
        print("=" * 60)
        print(f"\n  COST  (user: {d.cost.user_id})")
        print(f"    Daily spend : ${d.cost.daily_total_usd:.4f} / ${d.cost.daily_budget_usd:.2f}")
        print(f"    Remaining   : {d.cost.budget_remaining_pct:.1f}%")
        print(f"\n  RETRIEVAL QUALITY  ({d.retrieval.total_queries} queries)")
        print(f"    P@5         : {d.retrieval.precision_at_5:.4f}")
        print(f"    MRR         : {d.retrieval.mrr:.4f}")
        print(f"    NDCG        : {d.retrieval.ndcg:.4f}")
        print(f"    Avg chunks  : {d.retrieval.avg_chunks_retrieved}")
        if d.agent_call_counts:
            print(f"\n  AGENT CALLS")
            for agent, count in sorted(d.agent_call_counts.items()):
                print(f"    {agent:<12}: {count}")
        print(f"\n  Prometheus /metrics → :8001/metrics")
        print(f"  Grafana dashboard  → http://localhost:3000")
        print("=" * 60 + "\n")
