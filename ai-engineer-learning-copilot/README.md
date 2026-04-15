# AI Engineer Learning — Copilot Edition

A hands-on, notebook-driven curriculum for building production-grade AI systems using **100% free, local tools** — no paid APIs required.

Every concept is introduced in a Jupyter notebook that runs top-to-bottom using:
[Ollama](https://ollama.com/) · [sentence-transformers](https://www.sbert.net/) · [ChromaDB](https://www.trychroma.com/) · [LangChain](https://www.langchain.com/) · [dbt-duckdb](https://github.com/duckdb/dbt-duckdb)

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Start Ollama (Docker)
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama pull nomic-embed-text

# 4. Launch Jupyter Lab
jupyter lab
```

Open the `notebooks/` folder and work through the phases in numbered order.

---

## Project Structure

```
ai-engineer-learning-copilot/
├── notebooks/
│   ├── 01_embeddings/          # Phase 1 — Embeddings & Semantic Similarity
│   ├── 02_llms_and_prompting/  # Phase 2 — LLMs, Prompting & First RAG
│   ├── 03_rag_pipeline/        # Phase 3 — RAG Pipeline
│   ├── 04_advanced_rag/        # Phase 4 — Advanced RAG
│   └── 05_agents/              # Phase 5 — Agents
├── dbt_pipeline/
│   └── data/
│       └── raw/                # Raw data consumed by dbt models
├── data/
│   └── raw/                    # Raw data files (CSV, TXT, etc.)
├── requirements.txt
└── README.md
```

---

## Phases

### Phase 1 — Embeddings & Semantic Similarity (`notebooks/01_embeddings/`)

Build a deep intuitive understanding of how text is transformed into vectors.

- Load and explore a domain corpus (articles, regulatory documents)
- Compute dense embeddings with `sentence-transformers`
- Visualise high-dimensional embedding spaces with PCA / UMAP
- Measure cosine similarity and build a simple semantic search engine
- Understand why embedding quality is the single biggest lever in any RAG system

**Key tools:** `sentence-transformers`, `numpy`, `pandas`, `chromadb`

---

### Phase 2 — LLMs, Prompting & First RAG (`notebooks/02_llms_and_prompting/`)

Move from retrieval to generation and wire up your first end-to-end RAG pipeline.

- Run open-weight LLMs locally via Ollama (llama3.1:8b, nomic-embed-text)
- Master prompt engineering patterns: zero-shot, few-shot, chain-of-thought, structured output
- Build a document ingestion pipeline: load → chunk → embed → store in ChromaDB
- Create a retrieval-augmented generation (RAG) chain with LangChain
- Inspect retrieved context and generated answers side-by-side

**Key tools:** `ollama`, `langchain`, `langchain-community`, `langchain-chroma`, `chromadb`

---

### Phase 3 — RAG Pipeline (`notebooks/03_rag_pipeline/`)

Solidify the full RAG architecture and make it robust enough to evaluate.

- Refactor chunking strategy (size, overlap, metadata enrichment)
- Build a configurable retrieval pipeline (top-k, MMR, threshold filtering)
- Add a query-rewriting step to handle ambiguous user questions
- Introduce a basic evaluation loop using ground-truth question/answer pairs
- Generate a RAGAS evaluation report (faithfulness, answer relevancy, context recall)

**Key tools:** `langchain`, `ragas`, `chromadb`, `ollama`

---

### Phase 4 — Advanced RAG (`notebooks/04_advanced_rag/`)

Production RAG systems need more than dense retrieval alone.

- Implement hybrid search: BM25 sparse retrieval + dense vector retrieval
- Add a re-ranking step to improve precision at the top of the result list
- Explore parent-document retrieval and small-to-big chunk strategies
- Run systematic experiments and track results with a reproducible eval harness
- Connect the vector store refresh to a dbt pipeline run via Prefect

**Key tools:** `rank-bm25`, `langchain`, `ragas`, `prefect`, `dbt-duckdb`

---

### Phase 5 — Agents (`notebooks/05_agents/`)

Build stateful, multi-step agents that plan, use tools, and recover from errors.

- Introduce LangGraph and the concept of a stateful agent graph
- Give the agent tools: RAG retrieval, SQL over DuckDB, knowledge graph queries
- Load domain data into Neo4j and query it with Cypher
- Extract named entities from regulatory text to auto-populate the knowledge graph
- Add human-in-the-loop checkpoints and error-recovery logic

**Key tools:** `langchain`, `langchain-community`, `neo4j`, `prefect`, `ollama`

---

## Dependencies

All dependencies are pinned in `requirements.txt`.  A summary by role:

| Package | Role |
|---------|------|
| `sentence-transformers` | Local embedding models |
| `chromadb` | Local vector store |
| `langchain` + `langchain-community` + `langchain-chroma` | LLM orchestration |
| `ollama` | Local LLM inference |
| `jupyter` | Interactive notebooks |
| `numpy` / `pandas` | Numerical & tabular data |
| `rank-bm25` | Sparse BM25 retrieval |
| `ragas` | RAG evaluation metrics |
| `prefect` | Pipeline orchestration |
| `dbt-core` + `dbt-duckdb` | SQL data transformations |
| `neo4j` | Knowledge graph database |
| `faker` | Synthetic data generation |
| `requests` / `tqdm` | HTTP utilities & progress bars |

---

## Rules

1. Every notebook runs **top-to-bottom** with no manual steps.
2. Every code cell has a **markdown cell above it** explaining what it does, why it matters, and what to look for when you run it.
3. No paid APIs. No OpenAI keys. Everything runs locally.
4. Commit one file at a time with a descriptive commit message.

---

## License

MIT — see the root repository LICENSE for details.
