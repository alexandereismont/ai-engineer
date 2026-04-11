# Phase 05 — Agents & Knowledge Graphs

Weeks 19–23 · LangGraph · Neo4j · spaCy

---

## Overview

This phase builds a stateful AI agent that can answer pension regulation questions,
query fund metrics, and traverse a regulatory knowledge graph. The agent combines:

- **LangGraph** for multi-step, stateful reasoning with human-in-the-loop support
- **ChromaDB** (RAG) for regulation text retrieval
- **DuckDB** for fund metrics queries
- **Neo4j** for regulatory entity relationship traversal
- **spaCy** with custom EntityRuler for domain entity extraction

---

## Prerequisites

Before starting, ensure the following services are running and data is populated.

### 1. Neo4j

```bash
# Docker
docker run -d \
  --name neo4j-pension \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/pension_secret \
  neo4j:5

# Load the regulation graph
python knowledge_graph/neo4j_loader.py --clean
```

### 2. Ollama (LLM)

```bash
# Install Ollama from https://ollama.com/
ollama serve
ollama pull llama3.2
```

### 3. ChromaDB

```bash
# Docker
docker run -d --name chroma -p 8000:8000 chromadb/chroma:latest
```

Then populate it from Phase 02 or 03:

```bash
cd ../phase-03-advanced-rag
python ingest.py  # populates the "pension_docs" collection
```

### 4. DuckDB warehouse

Populated as part of Phase 04 dbt pipeline:

```bash
cd ../phase-04-pipelines
dbt run
```

### 5. spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 6. Python dependencies

```bash
pip install langgraph langchain-core langchain-ollama \
            chromadb duckdb neo4j \
            spacy
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `pension_secret` | Neo4j password |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8000` | ChromaDB port |
| `DUCKDB_PATH` | `../../data/pension_warehouse.duckdb` | DuckDB file path |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |

---

## How to start the agent

### Interactive CLI

```bash
cd phase-05-agents
python agent/pension_agent.py "What does IORP III say about the prudent person principle?"
```

### Python API

```python
from agent.pension_agent import run_agent, create_agent
from langchain_core.messages import HumanMessage

# Single query
answer = run_agent("What is the coverage ratio requirement under IORP III?")
print(answer)

# Multi-turn with memory
agent = create_agent(persist_memory=True)
config = {"configurable": {"thread_id": "session-001"}}

for query in [
    "What is IORP III?",
    "Which article covers ORSA?",
    "What are the reporting deadlines?",
]:
    state = {
        "messages": [HumanMessage(content=query)],
        "context": "",
        "tool_calls": 0,
        "requires_human": False,
    }
    result = agent.invoke(state, config=config)
    last_ai = next(m for m in reversed(result["messages"]) if hasattr(m, "content"))
    print(f"Q: {query}\nA: {last_ai.content}\n")
```

### Streaming

```python
from agent.pension_agent import create_agent
from langchain_core.messages import HumanMessage

agent = create_agent()
state = {
    "messages": [HumanMessage(content="Show me ABP fund metrics for 2023")],
    "context": "",
    "tool_calls": 0,
    "requires_human": False,
}

for chunk in agent.stream(state, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"[{node}] state update keys: {list(update.keys())}")
```

---

## Architecture

```
User Query
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│  LangGraph StateGraph                                             │
│                                                                   │
│  START ──► route_query                                            │
│                │                                                  │
│                ├── keyword: regulation/article/IORP/FTK ─────────►│
│                │         rag_retrieval (ChromaDB)                 │
│                │                   │                              │
│                ├── keyword: fund/return/AUM/year ────────────────►│
│                │         sql_query (DuckDB)                       │
│                │                   │                              │
│                ├── keyword: related/graph/entity/references ─────►│
│                │         graph_query (Neo4j)                      │
│                │                   │                              │
│                └── keyword: human/unsure/escalate ───────────────►│
│                          human_checkpoint ──► END (paused)        │
│                                    │                              │
│                                    ▼                              │
│                          generate_response (Ollama LLM)           │
│                                    │                              │
│                                   END                             │
└───────────────────────────────────────────────────────────────────┘
                │
    External integrations:
    ┌───────────┐  ┌─────────┐  ┌────────────────┐
    │ ChromaDB  │  │ DuckDB  │  │ Neo4j          │
    │ (pension_ │  │ (pension│  │ (regulatory    │
    │  docs)    │  │ _ware.. │  │  graph)        │
    └───────────┘  └─────────┘  └────────────────┘
```

### Knowledge graph schema

```
(Regulation)-[:CONTAINS]──►(Article)-[:REFERENCES]──►(Concept)
      │                         │
      └───[:DEFINES]───────────►│              (Article)-[:REQUIRES]──►(Requirement)
                                                                              │
                                                          (Requirement)-[:APPLIES_TO]──►(PensionFund)
```

---

## Example queries

| Category | Example query | Route taken |
|----------|--------------|-------------|
| Regulation | "What does IORP III say about the prudent person principle?" | `rag_retrieval` |
| Regulation | "What are the ORSA reporting deadlines under Article 14?" | `rag_retrieval` |
| Fund metrics | "Show me ABP fund performance for 2023" | `sql_query` |
| Fund metrics | "What is PFZW's coverage ratio and AUM in 2022?" | `sql_query` |
| Graph | "What concepts are referenced by Article 19?" | `graph_query` |
| Graph | "What articles are related to the ORSA requirement?" | `graph_query` |
| Escalation | "I'm confused about ESG reporting — can a human help?" | `human_checkpoint` |

---

## File structure

```
phase-05-agents/
├── agent/
│   ├── __init__.py
│   └── pension_agent.py          # LangGraph StateGraph agent
├── knowledge_graph/
│   ├── __init__.py
│   ├── neo4j_loader.py           # Schema + sample data loader
│   └── spacy_extractor.py        # Custom pension EntityRuler
├── notebooks/
│   ├── week1_langgraph_basics.ipynb
│   └── week2_neo4j_graph.ipynb
└── README.md
```

---

## Running the knowledge graph loader

```bash
# Load data (safe to re-run)
python knowledge_graph/neo4j_loader.py

# Wipe and reload
python knowledge_graph/neo4j_loader.py --clean

# Query after loading
python knowledge_graph/neo4j_loader.py --query "coverage_ratio"
```

## Running the spaCy extractor

```bash
python knowledge_graph/spacy_extractor.py
```

This prints extracted entities from three demo pension regulation sentences.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `OSError: Model 'en_core_web_sm' not found` | `python -m spacy download en_core_web_sm` |
| `ServiceUnavailable: Failed to establish connection to Neo4j` | Start Neo4j and check `NEO4J_URI` |
| `ChromaDB: Collection 'pension_docs' does not exist` | Run the Phase 02/03 ingest script |
| `DuckDB: Table 'dim_funds' not found` | Run the Phase 04 dbt pipeline |
| `Connection refused (Ollama)` | Run `ollama serve` and `ollama pull llama3.2` |
