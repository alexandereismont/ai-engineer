{% macro classify_coverage_ratio(column_name) %}
case
    when {{ column_name }} < 100    then 'underfunded'
    when {{ column_name }} < 110    then 'at_risk'
    when {{ column_name }} < 125    then 'healthy'
    else                                 'well_funded'
end
{% endmacro %}
