/*
  Silver Layer — employees_cleaned
  ──────────────────────────────────
  The silver layer applies business rules to produce clean, analysis-ready data.
  Transformations applied here:
    • Filter: drop rows where salary IS NULL — records without compensation data
      are incomplete and would skew aggregate calculations
    • Standardise: department names converted to lowercase so joins and GROUP BY
      are case-insensitive (avoids 'Engineering' ≠ 'engineering' duplicates)
    • Enrich: add tenure_years — the number of full years since hire_date,
      useful for segmenting workforce by seniority

  AI agent note: this is the table an LLM agent should query when answering
  "who works in engineering?" or "what is the average salary?". Gold-layer
  aggregations are derived from this model.
*/

{{ config(materialized='table') }}

select
    employee_id,
    name,
    lower(department)                                    as department,
    salary,
    hire_date,
    performance_score,
    date_diff('year', hire_date, current_date)           as tenure_years,
    loaded_at
from {{ ref('stg_employees') }}
where salary is not null
