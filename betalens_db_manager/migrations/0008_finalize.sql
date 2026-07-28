INSERT INTO betalens.dataset_coverage (logical_dataset)
SELECT logical_dataset
FROM unnest(ARRAY[
    'daily_market', 'daily_index', 'daily_fund', 'daily_bond',
    'fundamentals', 'macro', 'factors',
    'industry', 'index_universe', 'trade_status'
]) AS logical_dataset
ON CONFLICT (logical_dataset) DO NOTHING;

WITH all_rows AS (
    SELECT metric.logical_dataset,
           fact.trade_date + metric.availability_time AS available_at
    FROM betalens.market_daily_fact fact
    JOIN betalens.entity_dim entity ON entity.entity_id = fact.entity_id
    JOIN betalens.metric_dim metric
      ON metric.logical_dataset = CASE entity.entity_type
          WHEN 'stock' THEN 'daily_market'
          WHEN 'index' THEN 'daily_index'
          WHEN 'fund' THEN 'daily_fund'
          WHEN 'bond' THEN 'daily_bond'
      END
     AND metric.storage_kind = 'core'
    CROSS JOIN LATERAL (
        SELECT CASE metric.storage_column
            WHEN 'open' THEN fact.open
            WHEN 'high' THEN fact.high
            WHEN 'low' THEN fact.low
            WHEN 'close' THEN fact.close
            WHEN 'prev_close' THEN fact.prev_close
            WHEN 'volume' THEN fact.volume
            WHEN 'amount' THEN fact.amount
            WHEN 'turnover_rate' THEN fact.turnover_rate
        END AS value
    ) resolved
    WHERE resolved.value IS NOT NULL
    UNION ALL
    SELECT metric.logical_dataset, observation.available_at
    FROM betalens.observation_fact observation
    JOIN betalens.metric_dim metric ON metric.metric_id = observation.metric_id
    UNION ALL
    SELECT 'industry', membership.valid_from FROM betalens.industry_membership membership
    UNION ALL
    SELECT 'index_universe', snapshot.effective_at FROM betalens.index_snapshot snapshot
    UNION ALL
    SELECT 'trade_status', entity.first_trade_date + TIME '15:00:01'
    FROM betalens.entity_dim entity WHERE entity.first_trade_date IS NOT NULL
    UNION ALL
    SELECT 'trade_status', event.event_date + TIME '15:00:01'
    FROM betalens.trade_status_event event
), coverage_values AS (
    SELECT logical_dataset,
           min(available_at) AS min_at,
           max(available_at) AS max_at,
           count(*)::bigint AS row_count
    FROM all_rows
    GROUP BY logical_dataset
)
UPDATE betalens.dataset_coverage coverage
SET min_available_at = source.min_at,
    max_available_at = source.max_at,
    row_count = source.row_count,
    updated_at = clock_timestamp()
FROM coverage_values source
WHERE coverage.logical_dataset = source.logical_dataset;

COMMENT ON SCHEMA betalens IS
    'Betalens v2 physical schema. All DDL is owned by betalens_db_manager migrations.';
COMMENT ON TABLE betalens.schema_migration IS
    'Immutable applied migration versions and SHA-256 checksums.';
COMMENT ON TABLE betalens.entity_dim IS
    'Canonical security/economic-series identifiers; facts reference compact integer entity_id values.';
COMMENT ON TABLE betalens.entity_name_history IS
    'Point-in-time entity names. valid_to is an exclusive upper bound.';
COMMENT ON TABLE betalens.metric_dim IS
    'Canonical metric registry and physical storage routing.';
COMMENT ON TABLE betalens.metric_alias IS
    'Dataset-scoped aliases such as 成交额(元) -> 成交金额(元).';
COMMENT ON TABLE betalens.industry_scheme_dim IS
    'Industry classification systems such as 申万一级行业.';
COMMENT ON TABLE betalens.industry_dim IS
    'Industry codes and names scoped to one classification system.';
COMMENT ON TABLE betalens.market_daily_fact IS
    'Unpartitioned daily core market fact, one row per entity and trade date.';
COMMENT ON COLUMN betalens.market_daily_fact.remark IS
    'Per-metric legacy remarks keyed by canonical metric name.';
COMMENT ON TABLE betalens.observation_fact IS
    'Range-partitioned extended market, fundamental, macro, and factor observations.';
COMMENT ON COLUMN betalens.observation_fact.available_at IS
    'Earliest timestamp at which this value was available to a strategy.';
COMMENT ON COLUMN betalens.observation_fact.period_end IS
    'Optional theoretical/reporting period; it is not the point-in-time availability timestamp.';
COMMENT ON TABLE betalens.industry_membership IS
    'Point-in-time industry membership using [valid_from, valid_to) bounds.';
COMMENT ON TABLE betalens.index_snapshot IS
    'Point-in-time index-universe snapshot header.';
COMMENT ON TABLE betalens.index_constituent IS
    'Normalized constituents for an index snapshot.';
COMMENT ON TABLE betalens.trade_status_event IS
    'Sparse exceptional trade-status events; first normal date lives on entity_dim.';
COMMENT ON TABLE betalens.dataset_coverage IS
    'Observed time coverage and row-count metadata for logical datasets.';
