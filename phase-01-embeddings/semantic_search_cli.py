#!/usr/bin/env python
"""
Semantic Search CLI — Phase 01 Build Project
Usage: python semantic_search_cli.py --query "your query" [--top_k 5] [--category CATEGORY]

Builds or loads a pre-computed embedding index from ../data/raw/articles.csv,
then searches it semantically using cosine similarity.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'articles.csv')
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'article_embeddings.npy')
META_PATH = os.path.join(os.path.dirname(__file__), 'article_index_meta.csv')
MODEL_NAME = 'all-MiniLM-L6-v2'


def build_index(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Embed all abstracts and save to disk."""
    print(f"Building index for {len(df)} articles...")
    embeddings = model.encode(
        df['abstract'].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(INDEX_PATH, embeddings)
    df.to_csv(META_PATH, index=False)
    print(f"Index saved to {INDEX_PATH}")
    return embeddings


def load_or_build(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Load cached index or build it if not present."""
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("Loading cached index...")
        return np.load(INDEX_PATH)
    return build_index(df, model)


def search(
    query: str,
    corpus_embeddings: np.ndarray,
    df: pd.DataFrame,
    top_k: int = 5,
    category: str = None,
) -> pd.DataFrame:
    """Search the corpus for the most similar documents to the query."""
    model = SentenceTransformer(MODEL_NAME)
    query_emb = model.encode([query], normalize_embeddings=True)

    if category:
        mask = df['category'] == category
        filtered_idx = df.index[mask].tolist()
        filtered_embs = corpus_embeddings[filtered_idx]
        scores = (filtered_embs @ query_emb.T).flatten()
        top_local = np.argsort(scores)[::-1][:top_k]
        top_global = [filtered_idx[i] for i in top_local]
        top_scores = scores[top_local]
    else:
        scores = (corpus_embeddings @ query_emb.T).flatten()
        top_global = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_global]

    results = df.iloc[top_global][['id', 'title', 'category', 'year', 'abstract']].copy()
    results['score'] = top_scores
    results['score'] = results['score'].round(4)
    return results.reset_index(drop=True)


def print_results(results: pd.DataFrame, query: str) -> None:
    """Pretty-print search results."""
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")
    for i, row in results.iterrows():
        print(f"\n[{i+1}] Score: {row['score']:.4f} | {row['category']} | {row['year']}")
        print(f"    Title: {row['title']}")
        print(f"    {row['abstract'][:120]}...")
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Semantic Search CLI — search articles by meaning, not keywords.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python semantic_search_cli.py --query "pension fund solvency requirements"
  python semantic_search_cli.py --query "ESG investing" --top_k 10
  python semantic_search_cli.py --query "machine learning" --category general_ml
        """,
    )
    parser.add_argument('--query', required=True, help='Search query text')
    parser.add_argument('--top_k', type=int, default=5, help='Number of results (default: 5)')
    parser.add_argument('--category', default=None,
                        help='Filter by category: pension_regulation, investment_theory, '
                             'macroeconomics, fintech_ai, actuarial, general_ml')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild the index')
    args = parser.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    df = pd.read_csv(DATA_PATH)

    if args.rebuild and os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)

    corpus_embeddings = load_or_build(df, model)
    results = search(args.query, corpus_embeddings, df, args.top_k, args.category)
    print_results(results, args.query)


if __name__ == '__main__':
    main()
