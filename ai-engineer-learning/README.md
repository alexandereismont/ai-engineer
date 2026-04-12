# AI Engineer Learning Project

A hands-on, end-to-end learning curriculum for building production-grade AI systems.
Every exercise runs entirely on **free, local tools** — no paid API keys required.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull a local LLM (free — runs entirely on your machine)
ollama pull mistral        # or: ollama pull llama3  /  ollama pull phi3

# 3. Generate all datasets used across every notebook  <- run this first!
python data/generate_data.py

# 4. Launch Jupyter and open the first notebook
jupyter notebook notebooks/01_embeddings/
```

> **Why run the data script first?** All notebooks load files from
> `data/raw/`. The CSV files are excluded from git (they are generated
> artefacts, not source code). Running `generate_data.py` takes only a few
> seconds and produces fully deterministic output thanks to `random.seed(42)`.

---

## Prerequisites

- [Ollama](https://ollama.ai/) installed and running locally
- Python 3.10+

---

## Project Structure

```
ai-engineer-learning/
├── notebooks/
│   ├── 01_embeddings/          # Phase 1 — Embeddings & Vector Search
│   ├── 02_llms_and_prompting/  # Phase 2 — LLMs & Prompt Engineering
│   ├── 03_rag_pipeline/        # Phase 3 — RAG Pipeline
│   ├── 04_advanced_rag/        # Phase 4 — Advanced RAG
│   └── 05_agents/              # Phase 5 — Agents & Tool Use
├── dbt_pipeline/               # dbt-duckdb transformation pipeline
│   └── data/raw/               # Raw source data for dbt models
├── data/
│   └── raw/                    # Raw data used across notebooks
└── requirements.txt
```

---

## Curriculum Overview

### Phase 1 — Embeddings & Vector Search (`01_embeddings/`)

Learn how text is converted into numerical vectors, why semantic similarity
matters, and how to build and query a vector store from scratch.

**Topics covered:**
- What embeddings are and why they power modern AI search
- Generating embeddings with `sentence-transformers` (100% local)
- Storing and querying vectors in ChromaDB
- Comparing cosine similarity vs. BM25 keyword search
- Hybrid search: combining dense and sparse retrieval

**Key tools:** `sentence-transformers`, `chromadb`, `rank-bm25`, `numpy`

---

### Phase 2 — LLMs & Prompt Engineering (`02_llms_and_prompting/`)

Understand how large language models work, how to communicate with them
effectively, and how to structure prompts for reliable, reproducible outputs.

**Topics covered:**
- Running a local LLM via Ollama and the `ollama` Python client
- Prompt templates: zero-shot, few-shot, chain-of-thought
- System prompts and role-based messaging
- Output parsing and structured responses (JSON mode)
- Prompt versioning best practices

**Key tools:** `ollama`, `langchain`, `langchain-community`

---

### Phase 3 — RAG Pipeline (`03_rag_pipeline/`)

Build a complete Retrieval-Augmented Generation pipeline — the backbone of
most production AI applications.

**Topics covered:**
- Document loading, chunking strategies, and overlap
- Embedding documents into ChromaDB
- Retrieval: similarity search and MMR
- Prompt construction with retrieved context
- Evaluating answer quality with RAGAS metrics

**Key tools:** `langchain`, `langchain-chroma`, `chromadb`, `ragas`, `ollama`

---

### Phase 4 — Advanced RAG (`04_advanced_rag/`)

Level up the baseline RAG pipeline with techniques used in real production systems.

**Topics covered:**
- Query rewriting and HyDE
- Cross-encoder re-ranking
- Knowledge graph RAG with Neo4j
- Pipeline orchestration with Prefect
- Data transformation with dbt-duckdb

**Key tools:** `langchain`, `neo4j`, `prefect`, `dbt-core`, `dbt-duckdb`

---

### Phase 5 — Agents & Tool Use (`05_agents/`)

Build autonomous AI agents that plan, use tools, and complete multi-step tasks.

**Topics covered:**
- ReAct agent loop
- Defining and registering custom tools
- LangChain AgentExecutor patterns
- Memory: conversation history and summarization
- Multi-agent coordination

**Key tools:** `langchain`, `langchain-community`, `ollama`, `prefect`

---

## Generated Datasets (`data/raw/`)

All datasets are produced by `data/generate_data.py` using `random.seed(42)`.

| File | Rows | Used in |
|---|---|---|
| `articles.csv` | 200 | Phase 1, Phase 3 |
| `products.csv` | 500 | Phase 2, Phase 3 |
| `employees.csv` | 300 | Phase 4 (dbt) |
| `support_tickets.csv` | 400 | Phase 3 & 4 (RAG chatbot) |
| `knowledge_graph_nodes.csv` | 50 | Phase 4 (Graph RAG) |
| `knowledge_graph_edges.csv` | ~55 | Phase 4 (Graph RAG) |

All CSV files are in `.gitignore`. Regenerate with:

```bash
python data/generate_data.py
```

---

## Philosophy

> "I hear and I forget. I see and I remember. I do and I understand."

Every concept is taught through working code you can run, break, and modify.
