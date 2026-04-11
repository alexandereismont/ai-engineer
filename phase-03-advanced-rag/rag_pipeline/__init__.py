"""
rag_pipeline — production-quality RAG components for Phase 03.

Modules:
    config    — RAGConfig dataclass with all tuneable parameters
    indexer   — Indexer: load, chunk, embed and persist documents
    retriever — HybridRetriever: BM25 + vector + CrossEncoder reranking
    evaluator — RAGEvaluator: RAGAS-based evaluation and reporting
"""

from .config import RAGConfig
from .indexer import Indexer
from .retriever import HybridRetriever
from .evaluator import RAGEvaluator

__all__ = ["RAGConfig", "Indexer", "HybridRetriever", "RAGEvaluator"]
