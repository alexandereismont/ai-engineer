"""
indexer.py — Document loading, chunking, and index construction.

Supports five chunking strategies controlled by RAGConfig.chunking_strategy:
    fixed       — fixed character windows with overlap
    recursive   — LangChain RecursiveCharacterTextSplitter
    sentence    — NLTK sentence tokeniser, grouped to chunk_size
    semantic    — embedding-based boundary detection
    markdown    — header-aware splitting for structured documents
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import numpy as np

from .config import RAGConfig

logger = logging.getLogger(__name__)


class Indexer:
    """Load documents, chunk them, build vector and BM25 indexes."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def load_documents(self, paths: list[str]) -> list[dict]:
        """Load plain-text or CSV files and return a list of Document dicts.

        Each Document dict has keys: ``page_content``, ``metadata``.
        CSV files are expected to have at least an ``abstract`` column.
        """
        documents: list[dict] = []
        for path in paths:
            p = Path(path)
            if not p.exists():
                logger.warning("File not found, skipping: %s", path)
                continue

            if p.suffix == ".csv":
                documents.extend(self._load_csv(p))
            else:
                documents.extend(self._load_text(p))

        logger.info("Loaded %d documents from %d paths", len(documents), len(paths))
        return documents

    def _load_text(self, path: Path) -> list[dict]:
        text = path.read_text(encoding="utf-8")
        return [{"page_content": text, "metadata": {"source": str(path), "type": "text"}}]

    def _load_csv(self, path: Path) -> list[dict]:
        import pandas as pd

        df = pd.read_csv(path)
        text_col = "abstract" if "abstract" in df.columns else df.columns[0]
        docs = []
        for i, row in df.iterrows():
            content = str(row[text_col])
            metadata = {"source": str(path), "row": int(i), "type": "article"}
            if "title" in df.columns:
                metadata["title"] = str(row["title"])
            if "category" in df.columns:
                metadata["category"] = str(row["category"])
            docs.append({"page_content": content, "metadata": metadata})
        return docs

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """Split documents into chunks according to ``config.chunking_strategy``."""
        strategy = self.config.chunking_strategy
        logger.info("Chunking %d documents with strategy '%s'", len(documents), strategy)

        dispatch = {
            "fixed": self._chunk_fixed,
            "recursive": self._chunk_recursive,
            "sentence": self._chunk_sentence,
            "semantic": self._chunk_semantic,
            "markdown": self._chunk_markdown,
        }
        if strategy not in dispatch:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Choose from: {list(dispatch.keys())}"
            )
        chunks = dispatch[strategy](documents)
        logger.info("Produced %d chunks", len(chunks))
        return chunks

    def _chunk_fixed(self, documents: list[dict]) -> list[dict]:
        """Fixed-size character windowing with overlap."""
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        chunks = []
        for doc in documents:
            text = doc["page_content"]
            start = 0
            while start < len(text):
                end = start + size
                chunk_text = text[start:end]
                if chunk_text.strip():
                    chunks.append({
                        "page_content": chunk_text,
                        "metadata": {**doc["metadata"], "chunk_strategy": "fixed"},
                    })
                start += size - overlap
        return chunks

    def _chunk_recursive(self, documents: list[dict]) -> list[dict]:
        """LangChain RecursiveCharacterTextSplitter."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = []
        for doc in documents:
            splits = splitter.split_text(doc["page_content"])
            for split in splits:
                if split.strip():
                    chunks.append({
                        "page_content": split,
                        "metadata": {**doc["metadata"], "chunk_strategy": "recursive"},
                    })
        return chunks

    def _chunk_sentence(self, documents: list[dict]) -> list[dict]:
        """Group NLTK sentences until chunk_size is reached."""
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        size = self.config.chunk_size
        chunks = []
        for doc in documents:
            sentences = nltk.sent_tokenize(doc["page_content"])
            current: list[str] = []
            current_len = 0
            for sent in sentences:
                if current_len + len(sent) > size and current:
                    chunks.append({
                        "page_content": " ".join(current),
                        "metadata": {**doc["metadata"], "chunk_strategy": "sentence"},
                    })
                    current = []
                    current_len = 0
                current.append(sent)
                current_len += len(sent) + 1
            if current:
                chunks.append({
                    "page_content": " ".join(current),
                    "metadata": {**doc["metadata"], "chunk_strategy": "sentence"},
                })
        return chunks

    def _chunk_semantic(self, documents: list[dict]) -> list[dict]:
        """Embed consecutive sentences; start new chunk when cosine similarity drops."""
        import nltk
        from sentence_transformers import SentenceTransformer

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        threshold = 0.5
        model = SentenceTransformer("all-MiniLM-L6-v2")
        chunks = []

        for doc in documents:
            sentences = nltk.sent_tokenize(doc["page_content"])
            if len(sentences) < 2:
                chunks.append({
                    "page_content": doc["page_content"],
                    "metadata": {**doc["metadata"], "chunk_strategy": "semantic"},
                })
                continue

            embeddings = model.encode(sentences, show_progress_bar=False)
            current: list[str] = [sentences[0]]

            for i in range(1, len(sentences)):
                sim = float(
                    np.dot(embeddings[i - 1], embeddings[i])
                    / (np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-9)
                )
                if sim < threshold and current:
                    chunks.append({
                        "page_content": " ".join(current),
                        "metadata": {**doc["metadata"], "chunk_strategy": "semantic"},
                    })
                    current = []
                current.append(sentences[i])

            if current:
                chunks.append({
                    "page_content": " ".join(current),
                    "metadata": {**doc["metadata"], "chunk_strategy": "semantic"},
                })
        return chunks

    def _chunk_markdown(self, documents: list[dict]) -> list[dict]:
        """Split on Markdown headers (#, ##, ###) as natural boundaries."""
        import re

        header_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
        chunks = []

        for doc in documents:
            text = doc["page_content"]
            positions = [m.start() for m in header_re.finditer(text)]
            if not positions:
                # No headers found — fall back to recursive
                return self._chunk_recursive([doc])

            positions.append(len(text))
            for i in range(len(positions) - 1):
                section = text[positions[i]:positions[i + 1]].strip()
                if section:
                    chunks.append({
                        "page_content": section,
                        "metadata": {**doc["metadata"], "chunk_strategy": "markdown"},
                    })
        return chunks

    # ------------------------------------------------------------------
    # Parent-child chunking helper
    # ------------------------------------------------------------------

    def chunk_parent_child(
        self, documents: list[dict], child_size: int = 200, parent_size: int = 1000
    ) -> tuple[list[dict], list[dict]]:
        """Return (child_chunks, parent_chunks) for two-stage retrieval.

        Index child chunks for precision; store parent chunk id in metadata
        so the parent can be fetched for the LLM context window.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=100, separators=["\n\n", "\n", ". ", " "]
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=20, separators=["\n\n", "\n", ". ", " "]
        )

        parents: list[dict] = []
        children: list[dict] = []

        for doc in documents:
            parent_splits = parent_splitter.split_text(doc["page_content"])
            for p_idx, parent_text in enumerate(parent_splits):
                parent_id = f"{doc['metadata'].get('source', 'doc')}__p{p_idx}"
                parents.append({
                    "page_content": parent_text,
                    "metadata": {**doc["metadata"], "parent_id": parent_id},
                })
                child_splits = child_splitter.split_text(parent_text)
                for c_idx, child_text in enumerate(child_splits):
                    children.append({
                        "page_content": child_text,
                        "metadata": {
                            **doc["metadata"],
                            "parent_id": parent_id,
                            "child_idx": c_idx,
                        },
                    })

        return children, parents

    # ------------------------------------------------------------------
    # Vector index
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[dict]):
        """Embed chunks and persist a Chroma collection to disk.

        Returns the Chroma collection object.
        """
        import chromadb
        from chromadb.utils import embedding_functions

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        client = chromadb.PersistentClient(path=self.config.chroma_persist_dir)

        # Delete existing collection to avoid duplicate IDs on rebuild
        try:
            client.delete_collection(self.config.collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=self.config.collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            collection.add(
                ids=[f"chunk_{i + j}" for j in range(len(batch))],
                documents=[c["page_content"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )
            logger.debug("Indexed batch %d-%d", i, i + len(batch))

        logger.info(
            "Built Chroma index with %d chunks in '%s'",
            len(chunks),
            self.config.chroma_persist_dir,
        )
        return collection

    def load_index(self):
        """Load an existing Chroma collection from disk."""
        import chromadb
        from chromadb.utils import embedding_functions

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        client = chromadb.PersistentClient(path=self.config.chroma_persist_dir)
        collection = client.get_collection(
            name=self.config.collection_name, embedding_function=ef
        )
        logger.info(
            "Loaded Chroma index '%s' (%d items)",
            self.config.collection_name,
            collection.count(),
        )
        return collection

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def build_bm25_index(self, chunks: list[dict]):
        """Tokenise chunks and build a BM25Okapi index.

        Returns the BM25Okapi object; the caller should keep a reference to
        ``chunks`` for result lookup.
        """
        from rank_bm25 import BM25Okapi

        tokenised = [c["page_content"].lower().split() for c in chunks]
        index = BM25Okapi(tokenised)
        logger.info("Built BM25 index with %d documents", len(tokenised))
        return index
