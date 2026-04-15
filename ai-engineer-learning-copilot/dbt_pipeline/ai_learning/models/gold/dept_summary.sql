/*
  Gold Layer — dept_summary
  ──────────────────────────
  Business-ready aggregate for HR and management reporting.
  This model answers questions like:
    • "How many people are in engineering?"
    • "Which department pays the most?"
    • "Which team has the best performance?"

  AI agent use-case: an LLM with tool-use can call a SQL query against
  this table to answer natural-language HR questions without touching
  PII-sensitive individual rows in the silver layer.

  Columns:
    department        — standardised department name (from silver layer)
    headcount         — number of active employees with salary data
    avg_salary        — mean salary, rounded to 2 decimal places
    avg_performance   — mean performance score (1.0–5.0 scale)
    avg_tenure_years  — mean years of service
*/

{{ config(materialized='table') }}

select
    department,
    count(*)                             as headcount,
    round(avg(salary), 2)                as avg_salary,
    round(avg(performance_score), 2)     as avg_performance,
    round(avg(tenure_years), 1)          as avg_tenure_years
from {{ ref('employees_cleaned') }}
group by department
order by headcount desc
