/*
  Bronze Layer — stg_employees
  ─────────────────────────────
  This is the first layer of the medallion architecture. The bronze (staging)
  layer ingests raw seed data with minimal transformation:
    • Cast every column to its correct type
    • Add a metadata timestamp (loaded_at) so downstream models know when
      data arrived in the warehouse
    • No business logic — this layer must be a faithful reflection of the source

  Why? If something looks wrong downstream, you can always diff against the
  bronze model to tell whether the issue is in the source or in a transformation.
*/

{{ config(materialized='view') }}

select
    cast(employee_id      as integer)   as employee_id,
    cast(name             as varchar)   as name,
    cast(department       as varchar)   as department,
    cast(salary           as double)    as salary,
    cast(hire_date        as date)      as hire_date,
    cast(performance_score as double)   as performance_score,
    current_timestamp                   as loaded_at
from {{ ref('employees') }}
