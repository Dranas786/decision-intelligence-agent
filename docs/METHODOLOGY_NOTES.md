# Methodology Notes

The current quality and reporting flow is shaped by a few well-known patterns:

- Expectation-style data quality checks inspired by Great Expectations concepts such as schema, missingness, uniqueness, and rule validation.
- Source recency and freshness checks inspired by dbt freshness monitoring.
- Bronze, silver, and gold local reporting layers inspired by the medallion architecture pattern.
- Point-cloud defect views implemented as browser visuals using Plotly-style 3D scatter, heatmap, and line charts.

In this codebase the tools remain deterministic Python functions. The language model chooses tools and explains outputs, but it does not perform the calculations itself.
