# AI Engineer Learning Plan

> **Mål:** Gå fra ingen ML-bakgrunn til jobbklar AI Engineer, med fokus på roller som SPK og Avinor lyser ut.

---

## Hvordan bruke dette repoet

- Jobb deg gjennom notatbøkene **i rekkefølge** — hver bygger på den forrige
- Hvert notatbok = ett tema = ett git-commit
- Alle notatbøker har kjørbar kode — **kjør cellene selv**, ikke bare les
- Sett opp API-nøkler i `.env` (se [Oppsett](#oppsett) nedenfor) før du starter fase 2

---

## Faseoversikt

| Fase | Tema | Notatbøker | Nøkkelkompetanse |
|------|------|-----------|-----------------|
| 1 | Fundamenter | 01–03 | Python, data, ML-begreper |
| 2 | Kjerne AI-engineering | 04–13 | LLMs, RAG, agenter, MCP |
| 3 | Dataengineering | 14–16 | Pipelines, dbt, orkestrering |
| 4 | Sky & DevOps | 17–18 | Docker, Azure, CI/CD |
| 5 | Sikkerhet & governance | 19 | GDPR, ansvarlig AI |

**Estimert tid:** ~2–3 timer per notatbok, ca. 40–50 timer totalt.

---

## Notatbokoversikt

### Fase 1 — Fundamenter

| # | Fil | Tittel | Hva du lærer |
|---|-----|--------|-------------|
| 01 | `phase1-foundations/01_python_for_ai.ipynb` | Python for AI Engineers | List comprehensions, async/await, dataclasses, type hints, JSON-parsing |
| 02 | `phase1-foundations/02_numpy_pandas.ipynb` | Data: NumPy & Pandas | Arrays, vektorisert matematikk, DataFrames, datarensing |
| 03 | `phase1-foundations/03_ml_concepts_primer.ipynb` | ML-begreper uten matte | Trening vs inferens, tokens, transformers (konseptuelt), loss |

### Fase 2 — Kjerne AI-engineering

| # | Fil | Tittel | Hva du lærer |
|---|-----|--------|-------------|
| 04 | `phase2-core-ai/04_llm_fundamentals.ipynb` | Hvordan LLMs faktisk fungerer | Tokenisering (tiktoken), attention-intuisjon, kontekstvinduer, temperatur |
| 05 | `phase2-core-ai/05_llm_apis.ipynb` | Kalle LLMs: OpenAI, Anthropic, Azure | Chat-API, roller, streaming, tool calling, prompt engineering, kostnader |
| 06 | `phase2-core-ai/06_embeddings.ipynb` | Embeddings: tekst som tall | Embedding-vektorer, cosinus-likhet, chunking-strategier |
| 07 | `phase2-core-ai/07_vector_databases.ipynb` | Vektordatabaser | ANN-søk, ChromaDB, Qdrant, bygge søkbar kunnskapsbase |
| 08 | `phase2-core-ai/08_rag_basics.ipynb` | RAG fra bunnen av | Ingest→chunk→embed→lagre→hent→prompt→generer, uten rammeverk |
| 09 | `phase2-core-ai/09_rag_advanced.ipynb` | Avansert RAG: reranking & evaluering | BM25 hybrid-søk, cross-encoder reranking, HyDE, RAGAS-evaluering |
| 10 | `phase2-core-ai/10_ai_agents.ipynb` | AI-agenter: verktøy, minne, ReAct | ReAct-mønster, verktøyloop, 3 minnetyper |
| 11 | `phase2-core-ai/11_agent_frameworks.ipynb` | Agentrammeverk: LangGraph, AutoGen, CrewAI | Samme agent i 3 rammeverk, tilstandsmaskiner, rollebaserte agenter |
| 12 | `phase2-core-ai/12_mcp_and_a2a.ipynb` | MCP og agent-til-agent-protokoller | MCP server+klient, A2A-protokoll, to-agentpipeline |
| 13 | `phase2-core-ai/13_knowledge_graphs.ipynb` | Kunnskapsgrafer & GraphRAG | RDF-tripler, NetworkX, Neo4j, LLM-generert graf, multi-hop reasoning |

### Fase 3 — Dataengineering

| # | Fil | Tittel | Hva du lærer |
|---|-----|--------|-------------|
| 14 | `phase3-data-engineering/14_data_pipelines_etl.ipynb` | ETL-pipelines for AI | Extract/transform/load, Pydantic-validering, datakvalitet for LLMs |
| 15 | `phase3-data-engineering/15_dbt_and_snowflake.ipynb` | dbt & Snowflake for AI-team | dbt modeller/tester/docs (DuckDB), Snowflake Cortex SQL |
| 16 | `phase3-data-engineering/16_prefect_orchestration.ipynb` | Prefect: orkestrering av AI-pipelines | Flows, tasks, schedules, retries, betinget LLM-berikelse |

### Fase 4 — Sky & DevOps

| # | Fil | Tittel | Hva du lærer |
|---|-----|--------|-------------|
| 17 | `phase4-cloud-devops/17_docker_for_ai.ipynb` | Docker for AI Engineers | Dockerfile, multi-stage builds, docker-compose, GitHub Actions CI |
| 18 | `phase4-cloud-devops/18_azure_ai_foundry.ipynb` | Azure AI Foundry | AI Hub, Model Catalog, GPT-4o endpoint, managed identity, Key Vault |

### Fase 5 — Sikkerhet & Governance

| # | Fil | Tittel | Hva du lærer |
|---|-----|--------|-------------|
| 19 | `phase5-security-governance/19_ai_security_gdpr.ipynb` | AI-sikkerhet, GDPR & ansvarlig AI | GDPR for RAG, EU AI Act, prompt injection-forsvar, Guardrails AI |

---

## Oppsett

### 1. Installer Python-avhengigheter

Det anbefales å bruke et virtuelt miljø:

```bash
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
# eller: .venv\Scripts\activate  # Windows
```

Hver notatbok installerer sine egne avhengigheter med `%pip install` øverst. Ingen global `requirements.txt` trengs.

### 2. API-nøkler

Kopier `.env.example` til `.env` og fyll inn nøklene dine:

```bash
cp .env.example .env
```

Fase 2 (notatbok 05+) trenger minst én LLM-leverandør. Anthropic og OpenAI er de enkleste å komme i gang med.

---

## Kobling til jobbannonsene

| Tema | SPK-annonsen | Avinor-annonsen |
|------|-------------|-----------------|
| LLM-APIer (notatbok 05) | Azure OpenAI, AWS Bedrock | Azure OpenAI |
| Embeddings + vektorsøk (06–07) | Embedding-strategier, vektorsøk | Embedding-teknologier |
| RAG-systemer (08–09) | RAG-pipelines, reranking, Cortex Search | RAG-arkitektur |
| AI-agenter (10–12) | AI-agentsystem, MCP-servere | Agentrammeverk, MCP, A2A |
| Kunnskapsgrafer (13) | Kunnskapsgraf, semantiske modeller | — |
| Datapipelines (14–16) | dbt, Snowflake, Prefect | — |
| Docker + CI/CD (17) | GitHub Actions | Docker, DevOps, CI/CD |
| Azure (18) | Azure OpenAI | Azure AI Foundry |
| GDPR + sikkerhet (19) | GDPR, offentlig sektor compliance | Sikkerhetsmekanismer, logging |
