/*
  Gold Layer — product_category_summary
  ──────────────────────────────────────
  Business-ready aggregate for product management and merchandising.
  Answers questions like:
    • "How many products do we sell in each category?"
    • "Which category has the best average rating?"
    • "How much review activity do Electronics generate?"

  AI agent use-case: an LLM agent querying this model can answer
  category-level product questions without scanning every product row.

  Columns:
    category           — product category name
    product_count      — number of products with valid price (from silver)
    avg_price          — mean price, rounded to 2 decimal places
    avg_rating         — mean customer rating (1.0–5.0 scale)
    total_reviews      — total review volume across all products in category
*/

{{ config(materialized='table') }}

select
    category,
    count(*)                             as product_count,
    round(avg(price), 2)                 as avg_price,
    round(avg(rating), 2)                as avg_rating,
    sum(review_count)                    as total_reviews
from {{ ref('products_cleaned') }}
group by category
order by product_count desc
