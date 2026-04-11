-- Custom test: fails if any coverage ratio is negative or zero
-- A coverage ratio ≤ 0 indicates a data quality error in the source
select
    fund_id,
    fund_name,
    reporting_year,
    coverage_ratio_pct
from {{ ref('stg_pension_funds') }}
where coverage_ratio_pct <= 0
