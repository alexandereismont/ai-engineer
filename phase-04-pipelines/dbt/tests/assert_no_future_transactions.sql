-- Custom test: fails if any transaction date is in the future
-- Catches data ingestion errors (wrong year format, etc.)
select
    transaction_id,
    fund_id,
    transaction_date
from {{ ref('stg_transactions') }}
where transaction_date > current_date
