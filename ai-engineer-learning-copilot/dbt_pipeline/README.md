# dbt Pipeline — ai_learning

This is the data engineering layer of the AI Engineer learning project.
It uses **dbt + DuckDB** to transform raw CSV data through a three-layer
medallion architecture (Bronze → Silver → Gold).

## Project layout

```
dbt_pipeline/
├── ai_learning/           ← dbt project root
│   ├── dbt_project.yml    ← project config (layers, materialisation settings)
│   ├── profiles.yml       ← connection config (DuckDB path)
│   ├── seeds/
│   │   ├── employees.csv  ← raw employee data (20 rows)
│   │   └── products.csv   ← raw product catalogue (20 rows)
│   └── models/
│       ├── schema.yml     ← all model/column docs + tests
│       ├── bronze/        ← staging views: cast types, add loaded_at
│       │   ├── stg_employees.sql
│       │   └── stg_products.sql
│       ├── silver/        ← cleaned tables: filter bad data, enrich
│       │   ├── employees_cleaned.sql
│       │   └── products_cleaned.sql
│       └── gold/          ← business aggregates for reporting / AI agents
│           ├── dept_summary.sql
│           └── product_category_summary.sql
└── data/                  ← DuckDB file lives here (gitignored)
```

## Architecture

```
CSV Seeds  →  Bronze (views)  →  Silver (tables)  →  Gold (tables)
              cast + stamp        clean + enrich       aggregate
```

| Layer   | Materialisation | Purpose |
|---------|----------------|---------|
| Bronze  | View            | Type-cast raw data, add `loaded_at` timestamp |
| Silver  | Table           | Remove bad rows, standardise, add derived columns |
| Gold    | Table           | Business-ready aggregates — what AI agents query |

## How to run

All commands should be run from inside `ai_learning/`:

```bash
cd ai_learning/
```

### 1. Load seed data

```bash
dbt seed --profiles-dir .
```

Loads `employees.csv` and `products.csv` into DuckDB.

### 2. Build all models

```bash
dbt run --profiles-dir .
```

Runs all 6 models in dependency order:
`stg_employees` → `employees_cleaned` → `dept_summary`
`stg_products`  → `products_cleaned`  → `product_category_summary`

### 3. Run tests

```bash
dbt test --profiles-dir .
```

Runs 51 data quality tests:
- `not_null` on every primary key and required column
- `unique` on every primary key
- `accepted_values` on `department`, `status`, and `price_tier`

### 4. Generate and serve documentation

```bash
dbt docs generate --profiles-dir . && dbt docs serve --profiles-dir .
```

Opens an interactive data catalogue in your browser at `http://localhost:8080`.
The lineage graph shows the full Bronze → Silver → Gold dependency chain.

## Connection

The DuckDB file is written to `../data/ai_learning.duckdb` (relative to the
project root). This path can be overridden with the `DBT_DUCKDB_PATH`
environment variable:

```bash
DBT_DUCKDB_PATH=/custom/path/mydb.duckdb dbt run --profiles-dir .
```

## AI agent integration (Phase 05)

The Gold layer tables are the natural interface between this pipeline and the
LLM agents built in Phase 05. An agent with SQL tool-use can query:

- `main_gold.dept_summary` — to answer "how many people are in engineering?"
- `main_gold.product_category_summary` — to answer "which category has the best rating?"

Because the gold layer is pre-aggregated, agents never need to scan individual
employee or product rows, keeping queries fast and avoiding PII exposure.
