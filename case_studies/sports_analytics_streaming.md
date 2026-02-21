# Sports Analytics Streaming (NBA Pipeline Analogy)

## Scenario
Near-real-time event scoring for player impact streams.

## System design
- Streaming inference using JSONL event processing in `real_time_pipelines/unified_pipeline.py`.
- Batch backfills for historical recalibration.
- SQL warehouse views (`warehouse_star_schema.sql`, `student_kpi_queries.sql`) adapted as a sports analytics data mart pattern.

## Trade-offs
- Micro-batching improves throughput but adds per-event delay.
- Feature freshness competes with latency budgets.
