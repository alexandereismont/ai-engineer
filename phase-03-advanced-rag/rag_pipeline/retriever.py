"""
retriever.py — Hybrid BM25 + vector retrieval with CrossEncoder reranking.

Two-stage pipeline:
    Stage 1 (recall)   — BM25 top-k  +  vector top-k  fused with RRF
    Stage 2 (precision)— CrossEncoder reranking of the fused candidates
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from .config import RAGConfig

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine BM25 and dense vector search with optional CrossEncoder reranking.

    Parameters
    ----------
    config:
        RAGConfig instance controlling retrieval behaviour.
    chroma_store:
        A Chroma collection (returned by Indexer.build_index / load_index).
    bm25_index:
        A BM25Okapi index (returned by Indexer.build_bm25_index).
    chunks_text:
        List of raw chunk strings in the same order as the BM25 index.
    """

    def __init__(
        self,
        config: RAGConfig,
        chroma_store,
        bm25_index,
        chunks_text: list[str],
    ) -> None:
        self.config = config
        self.chroma = chroma_store
        self.bm25 = bm25_index
        self.chunks_text = chunks_text
        self._cross_encoder = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[dict]:
        """Return final top-k chunks for *query*.

        Pipeline:
            1. BM25 top-(initial_retrieval_k)
            2. Dense vector top-(initial_retrieval_k)
            3. RRF merge
            4. (optional) CrossEncoder reranking
        """
        k = self.config.initial_retrieval_k

        bm25_ids = self._bm25_search(query, top_k=k)
        vector_ids = self._vector_search(query, top_k=k)

        if self.config.use_bm25:
            fused_ids = self._rrf_merge([bm25_ids, vector_ids], k=self.config.rrf_k)
        else:
            fused_ids = vector_ids

        candidates = [
            {"page_content": self.chunks_text[i], "metadata": {"chunk_index": i}}
            for i in fused_ids[: self.config.initial_retrieval_k]
            if i < len(self.chunks_text)
        ]

        if self.config.use_reranking and candidates:
            candidates = self._rerank(query, candidates)

        final = candidates[: self.config.final_top_k]
        logger.debug(
            "retrieve('%s'): bm25=%d, vector=%d, fused=%d, final=%d",
            query[:60],
            len(bm25_ids),
            len(vector_ids),
            len(fused_ids),
            len(final),
        )
        return final

    # ------------------------------------------------------------------
    # Stage 1: individual rankers
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> list[int]:
        """Return doc indices sorted by BM25 score (descending)."""
        tokenised_query = query.lower().split()
        scores = self.bm25.get_scores(tokenised_query)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [int(i) for i in ranked]

    def _vector_search(self, query: str, top_k: int) -> list[int]:
        """Return Chroma doc indices sorted by cosine similarity (descending).

        Chroma stores chunk IDs as "chunk_<idx>"; we parse the integer back out.
        If a BGE embedding prefix is configured it is prepended to the query only
        (not to documents, per BGE guidance).
        """
        effective_query = query
        if self.config.embedding_prefix and "bge" in self.config.embedding_model.lower():
            effective_query = self.config.embedding_prefix + query

        results = self.chroma.query(
            query_texts=[effective_query],
            n_results=min(top_k, self.chroma.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        indices = []
        for doc_id in ids:
            try:
                idx = int(doc_id.split("_")[1])
                indices.append(idx)
            except (IndexError, ValueError):
                logger.warning("Could not parse chunk index from id '%s'", doc_id)
        return indices

    # ------------------------------------------------------------------
    # Stage 1.5: RRF fusion
    # ------------------------------------------------------------------

    def _rrf_merge(self, rankings: list[list[int]], k: int = 60) -> list[int]:
        """Reciprocal Rank Fusion across multiple ranked lists.

        score(d) = sum_r  1 / (k + rank(d, r) + 1)
        """
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # ------------------------------------------------------------------
    # Stage 2: CrossEncoder reranking
    # ------------------------------------------------------------------

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Score query-document pairs with a CrossEncoder; return sorted docs."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder model '%s'", self.config.reranking_model)
            self._cross_encoder = CrossEncoder(self.config.reranking_model)

        pairs = [[query, c["page_content"]] for c in candidates]
        scores = self._cross_encoder.predict(pairs)

        scored = sorted(
            zip(scores, candidates), key=lambda x: x[0], reverse=True
        )
        reranked = []
        for score, doc in scored:
            doc = {**doc, "metadata": {**doc.get("metadata", {}), "rerank_score": float(score)}}
            reranked.append(doc)
        return reranked
