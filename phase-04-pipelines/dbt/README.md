# Pension Pipeline — dbt Project

## Purpose

Transform raw CSVs into a clean analytical warehouse using the **Medallion architecture**. Raw source data (pension funds, transactions, articles) is ingested into DuckDB and progressively refined through three layers until it reaches gold-quality analytical tables and a RAG-ready article corpus.

---

## Architecture

```
Bronze (staging)  →  Silver (intermediate)  →  Gold (marts)
─────────────────    ──────────────────────    ───────────────────
stg_pension_funds    int_fund_yearly_metrics   dim_funds
stg_transactions     int_transaction_enriched  fact_transactions
stg_articles                                   mart_rag_corpus
```

| Layer | dbt Materialization | Purpose |
|-------|---------------------|---------|
| Bronze | `view` (staging schema) | Cast types, standardize names, pseudonymize PII |
| Silver | `ephemeral` | Enrich and join; never persisted on its own |
| Gold | `table` (marts schema) | Final analytical tables; consumed by agents & dashboards |

---

## Setup

### 1. Install dependencies

```bash
pip install dbt-duckdb
```

### 2. Install dbt packages

```bash
cd phase-04-pipelines/dbt
dbt deps
```

### 3. Load raw CSV data into DuckDB

The raw CSVs live in `data/raw/`. Load them into DuckDB before running dbt:

```python
import duckdb

con = duckdb.connect("data/pension_warehouse.duckdb")

# Load each CSV as a table in the main schema
con.execute("""
    CREATE OR REPLACE TABLE pension_funds AS
    SELECT * FROM read_csv_auto('data/raw/pension_funds.csv', header=true);
""")

con.execute("""
    CREATE OR REPLACE TABLE transactions AS
    SELECT * FROM read_csv_auto('data/raw/transactions.csv', header=true);
""")

con.execute("""
    CREATE OR REPLACE TABLE articles AS
    SELECT * FROM read_csv_auto('data/raw/articles.csv', header=true);
""")

con.close()
print("Raw data loaded into DuckDB.")
```

---

## Running the Pipeline

```bash
# Run all models
dbt run

# Run only a specific layer
dbt run --select staging
dbt run --select marts

# Run tests
dbt test

# Seed lookup tables
dbt seed

# Generate and serve documentation
dbt docs generate && dbt docs serve
```

---

## Model Descriptions

| Model | Layer | Materialization | Description |
|-------|-------|-----------------|-------------|
| `stg_pension_funds` | Bronze | View | Typed, cleaned pension fund annual metrics; adds `coverage_status` |
| `stg_transactions` | Bronze | View | Cleaned transactions; PII fields (email, IBAN) replaced with MD5 hashes |
| `stg_articles` | Bronze | View | Filtered article corpus with quality flag for RAG eligibility |
| `int_fund_yearly_metrics` | Silver | Ephemeral | Fund metrics joined with per-year transaction aggregates |
| `int_transaction_enriched` | Silver | Ephemeral | Transactions enriched with fund name, country, and regulatory regime |
| `dim_funds` | Gold | Table | Fund dimension — one row per fund at latest reporting year |
| `fact_transactions` | Gold | Table | PII-free transaction fact table with fund context |
| `mart_rag_corpus` | Gold | Table | Quality-scored article corpus; bridges dbt pipeline to Chroma vector store |

---

## Data Lineage

```
raw.pension_funds ──► stg_pension_funds ──► int_fund_yearly_metrics ──► dim_funds
                                        └──────────────────────────────►
raw.transactions  ──► stg_transactions  ──► int_fund_yearly_metrics
                                        └──► int_transaction_enriched ──► fact_transactions

raw.articles      ──► stg_articles ──────────────────────────────────► mart_rag_corpus
                                                                              │
                                                                              ▼
                                                                    Chroma vector store
                                                                    (Phase 02/03 RAG)
```

`mart_rag_corpus` is the **key bridge** between this dbt pipeline and the Chroma vector store used by the Phase 05 agent. After each `dbt run`, the Prefect pipeline re-indexes Chroma with the freshly scored articles.
