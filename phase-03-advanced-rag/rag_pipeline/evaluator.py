"""
evaluator.py — RAGAS-based evaluation and reporting for the production RAG pipeline.

Metrics evaluated:
    faithfulness       — fraction of answer claims verifiable from context
    answer_relevancy   — embedding similarity between question and answer
    context_precision  — ratio of relevant retrieved chunks to total retrieved
    context_recall     — recall of relevant content vs ground-truth answer
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RAGConfig

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate a RAG pipeline with RAGAS metrics and produce human-readable reports.

    Parameters
    ----------
    config:
        RAGConfig instance; ``eval_dataset_path`` points to golden_dataset.json.
    """

    # RAGAS >= 0.1 metric defaults used as "strong" thresholds
    STRONG_THRESHOLDS: dict[str, float] = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.80,
        "context_precision": 0.70,
        "context_recall": 0.70,
    }

    def __init__(self, config: RAGConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def load_golden_dataset(self) -> list[dict]:
        """Load and validate the golden dataset JSON file.

        Expected schema per item::

            {
                "question":    str,
                "ground_truth": str,
                "contexts":    list[str],
                "source":      str   # "regulation" | "ips"
            }
        """
        path = Path(self.config.eval_dataset_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Golden dataset not found at '{path}'. "
                "Check RAGConfig.eval_dataset_path."
            )
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        required_keys = {"question", "ground_truth", "contexts"}
        for i, item in enumerate(data):
            missing = required_keys - item.keys()
            if missing:
                raise ValueError(
                    f"Item {i} in golden dataset is missing keys: {missing}"
                )
        logger.info("Loaded %d items from golden dataset at '%s'", len(data), path)
        return data

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, qa_chain, dataset: list[dict]) -> dict[str, float]:
        """Run RAGAS evaluation and return a dict of metric name → score.

        Parameters
        ----------
        qa_chain:
            A callable ``qa_chain(question: str) -> dict`` that must return at
            minimum ``{"result": str, "source_documents": list}``.
        dataset:
            Output of :meth:`load_golden_dataset`.

        Returns
        -------
        dict[str, float]
            Keys: faithfulness, answer_relevancy, context_precision, context_recall.
        """
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        logger.info("Running RAG inference on %d questions ...", len(dataset))

        rows: list[dict[str, Any]] = []
        for item in dataset:
            question = item["question"]
            ground_truth = item["ground_truth"]
            reference_contexts = item["contexts"]

            try:
                result = qa_chain(question)
                answer = result.get("result", "")
                source_docs = result.get("source_documents", [])
                retrieved_contexts = [
                    d.page_content if hasattr(d, "page_content") else str(d)
                    for d in source_docs
                ]
            except Exception as exc:
                logger.warning("QA chain failed for question '%s': %s", question[:60], exc)
                answer = ""
                retrieved_contexts = reference_contexts  # fallback

            rows.append({
                "question": question,
                "answer": answer,
                "contexts": retrieved_contexts if retrieved_contexts else reference_contexts,
                "ground_truth": ground_truth,
            })

        hf_dataset = Dataset.from_list(rows)

        logger.info("Running RAGAS evaluation ...")
        results = ragas_evaluate(
            hf_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        metrics: dict[str, float] = {}
        for metric_name in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]:
            value = results[metric_name]
            metrics[metric_name] = float(value) if value is not None else 0.0

        logger.info("RAGAS evaluation complete: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(
        self,
        metrics: dict[str, float],
        baseline: dict[str, float] | None = None,
    ) -> None:
        """Print a formatted table of RAGAS metrics to stdout.

        If *baseline* is provided (e.g. Week 6 scores), a delta column is shown.
        """
        metric_labels = {
            "faithfulness": "Faithfulness       ",
            "answer_relevancy": "Answer Relevancy   ",
            "context_precision": "Context Precision  ",
            "context_recall": "Context Recall     ",
        }

        sep = "─" * 68
        print(f"\n{sep}")
        print("  RAGAS Evaluation Report")
        print(sep)

        if baseline:
            print(f"  {'Metric':<22}  {'Score':>8}  {'Baseline':>9}  {'Delta':>7}  {'Status'}")
            print(sep)
            for key, label in metric_labels.items():
                score = metrics.get(key, 0.0)
                base = baseline.get(key, 0.0)
                delta = score - base
                threshold = self.STRONG_THRESHOLDS[key]
                status = "PASS" if score >= threshold else "FAIL"
                delta_str = f"{delta:+.3f}"
                print(
                    f"  {label}  {score:>8.3f}  {base:>9.3f}  {delta_str:>7}  {status}"
                )
        else:
            print(f"  {'Metric':<22}  {'Score':>8}  {'Threshold':>10}  {'Status'}")
            print(sep)
            for key, label in metric_labels.items():
                score = metrics.get(key, 0.0)
                threshold = self.STRONG_THRESHOLDS[key]
                status = "PASS" if score >= threshold else "FAIL"
                print(f"  {label}  {score:>8.3f}  {threshold:>10.3f}  {status}")

        avg = sum(metrics.values()) / len(metrics) if metrics else 0.0
        print(sep)
        print(f"  {'Average':<22}  {avg:>8.3f}")
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        metrics: dict[str, float],
        output_path: str | None = None,
    ) -> str:
        """Save evaluation results as timestamped JSON.

        Parameters
        ----------
        metrics:
            Dict returned by :meth:`evaluate`.
        output_path:
            If *None*, saves to ``./ragas_results_<timestamp>.json``.

        Returns
        -------
        str
            Absolute path of the saved file.
        """
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if output_path is None:
            output_path = f"./ragas_results_{ts}.json"

        payload = {
            "timestamp": ts,
            "embedding_model": self.config.embedding_model,
            "chunking_strategy": self.config.chunking_strategy,
            "chunk_size": self.config.chunk_size,
            "use_bm25": self.config.use_bm25,
            "use_reranking": self.config.use_reranking,
            "reranking_model": self.config.reranking_model,
            "metrics": metrics,
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        logger.info("Saved RAGAS results to '%s'", path.resolve())
        return str(path.resolve())
