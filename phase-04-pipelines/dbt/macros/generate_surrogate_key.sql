{% macro generate_surrogate_key(field_list) %}
    md5(
        cast(
            {% for field in field_list %}
                coalesce(cast({{ field }} as varchar), '_dbt_utils_surrogate_key_null_')
                {% if not loop.last %} || '-' || {% endif %}
            {% endfor %}
        as varchar)
    )
{% endmacro %}
