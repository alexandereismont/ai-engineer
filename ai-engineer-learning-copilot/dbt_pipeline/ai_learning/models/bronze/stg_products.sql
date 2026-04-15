/*
  Bronze Layer — stg_products
  ────────────────────────────
  Staging model for the raw products seed file. Responsibilities:
    • Cast all columns to their declared types
    • Stamp every row with a loaded_at timestamp for data lineage
    • Preserve all source rows — including dirty data (negative prices,
      discontinued items) so that the silver layer can make intentional
      decisions about what to keep or drop

  Rule of thumb: never silently discard rows in bronze. Make filtering
  explicit and visible in the silver layer.
*/

{{ config(materialized='view') }}

select
    cast(product_id    as integer)   as product_id,
    cast(name          as varchar)   as name,
    cast(category      as varchar)   as category,
    cast(price         as double)    as price,
    cast(rating        as double)    as rating,
    cast(review_count  as integer)   as review_count,
    cast(status        as varchar)   as status,
    current_timestamp                as loaded_at
from {{ ref('products') }}
