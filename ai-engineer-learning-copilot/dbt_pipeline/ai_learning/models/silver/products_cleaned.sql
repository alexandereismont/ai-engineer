/*
  Silver Layer — products_cleaned
  ────────────────────────────────
  Business rules applied to the raw products data:
    • Filter: remove rows where price <= 0 — these represent data-entry errors
      or placeholder records; they must never appear in pricing analysis
    • Enrich: add price_tier to bucket products into human-readable segments:
        budget   → price < 25
        mid      → price between 25 and 100 (inclusive)
        premium  → price > 100
    • Round rating to 1 decimal place for consistent display

  The price_tier column is a good example of a business rule that belongs in
  dbt, not in application code — it is defined once and reused everywhere.
*/

{{ config(materialized='table') }}

select
    product_id,
    name,
    category,
    price,
    round(rating, 1)                                     as rating,
    review_count,
    status,
    case
        when price < 25    then 'budget'
        when price <= 100  then 'mid'
        else                    'premium'
    end                                                  as price_tier,
    loaded_at
from {{ ref('stg_products') }}
where price > 0
