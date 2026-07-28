CREATE OR REPLACE FUNCTION betalens._migrate_legacy_long(
    p_dataset TEXT,
    p_entity_type TEXT
) RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    source_table REGCLASS;
    source_kind "char";
    conflicting_alias_key BOOLEAN;
BEGIN
    source_table := to_regclass(format('public.%I', p_dataset));
    IF source_table IS NULL THEN
        RETURN;
    END IF;
    SELECT relkind INTO source_kind FROM pg_class WHERE oid = source_table;
    IF source_kind NOT IN ('r', 'p') THEN
        RETURN;
    END IF;

    EXECUTE format($query$
        INSERT INTO betalens.entity_dim (code, entity_type, current_name)
        SELECT DISTINCT ON (code)
               code,
               %L,
               COALESCE(NULLIF(name, ''), code)
        FROM %s
        WHERE code IS NOT NULL AND code <> ''
        ORDER BY code, datetime DESC
        ON CONFLICT (code) DO UPDATE
        SET current_name = EXCLUDED.current_name,
            entity_type = CASE
                WHEN betalens.entity_dim.entity_type = 'unknown' THEN EXCLUDED.entity_type
                ELSE betalens.entity_dim.entity_type
            END,
            updated_at = clock_timestamp()
    $query$, p_entity_type, source_table);

    EXECUTE format($query$
        INSERT INTO betalens.metric_dim
            (logical_dataset, metric_name, storage_kind, storage_column, availability_time)
        SELECT DISTINCT %L, trim(source.metric), 'observation', NULL, TIME '15:00:01'
        FROM %s source
        LEFT JOIN betalens.metric_alias alias
          ON alias.logical_dataset = %L AND alias.alias = trim(source.metric)
        LEFT JOIN betalens.metric_dim direct
          ON direct.logical_dataset = %L AND direct.metric_name = trim(source.metric)
        WHERE source.metric IS NOT NULL
          AND trim(source.metric) <> ''
          AND alias.metric_id IS NULL
          AND direct.metric_id IS NULL
        ON CONFLICT (logical_dataset, metric_name) DO NOTHING
    $query$, p_dataset, source_table, p_dataset, p_dataset);

    INSERT INTO betalens.metric_alias (logical_dataset, alias, metric_id)
    SELECT metric.logical_dataset, metric.metric_name, metric.metric_id
    FROM betalens.metric_dim metric
    WHERE metric.logical_dataset = p_dataset
    ON CONFLICT (logical_dataset, alias) DO NOTHING;

    EXECUTE format($query$
        SELECT EXISTS (
            SELECT 1
            FROM %s source
            JOIN betalens.metric_alias alias
              ON alias.logical_dataset = %L AND alias.alias = trim(source.metric)
            GROUP BY source.datetime, source.code, alias.metric_id
            HAVING count(DISTINCT COALESCE(source.value::text, '<NULL>')) > 1
                OR count(DISTINCT COALESCE(source.name, '<NULL>')) > 1
                OR count(DISTINCT COALESCE(source.remark, 'null'::jsonb)) > 1
        )
    $query$, source_table, p_dataset)
    INTO conflicting_alias_key;
    IF conflicting_alias_key THEN
        RAISE EXCEPTION
            '旧表 public.% 中多个指标别名解析到同一物理 key，但值/name/remark 不一致',
            p_dataset;
    END IF;

    EXECUTE format($query$
        WITH timeline AS (
            SELECT DISTINCT ON (entity.entity_id, source.datetime)
                   entity.entity_id,
                   source.datetime::timestamp AS valid_from,
                   source.name
            FROM %s source
            JOIN betalens.entity_dim entity ON entity.code = source.code
            WHERE source.datetime IS NOT NULL AND source.name IS NOT NULL AND source.name <> ''
            ORDER BY entity.entity_id, source.datetime, source.metric
        ), marked AS (
            SELECT timeline.*,
                   lag(name) OVER (PARTITION BY entity_id ORDER BY valid_from) AS previous_name
            FROM timeline
        )
        INSERT INTO betalens.entity_name_history (entity_id, valid_from, valid_to, name)
        SELECT entity_id, valid_from, NULL, name
        FROM marked
        WHERE previous_name IS DISTINCT FROM name
        ON CONFLICT (entity_id, valid_from) DO UPDATE SET name = EXCLUDED.name
    $query$, source_table);

    IF p_dataset IN ('daily_market', 'daily_index', 'daily_fund', 'daily_bond') THEN
        EXECUTE format($query$
            WITH resolved AS (
                SELECT entity.entity_id,
                       source.datetime::timestamp AS available_at,
                       source.value::double precision AS value,
                       source.remark,
                       metric.metric_name,
                       metric.storage_column
                FROM %s source
                JOIN betalens.entity_dim entity ON entity.code = source.code
                JOIN betalens.metric_alias alias
                  ON alias.logical_dataset = %L AND alias.alias = trim(source.metric)
                JOIN betalens.metric_dim metric ON metric.metric_id = alias.metric_id
                WHERE metric.storage_kind = 'core'
                  AND source.value IS NOT NULL
                  AND source.datetime::time = metric.availability_time
            ), wide AS (
                SELECT entity_id,
                       available_at::date AS trade_date,
                       max(value) FILTER (WHERE storage_column = 'open') AS open,
                       max(value) FILTER (WHERE storage_column = 'high') AS high,
                       max(value) FILTER (WHERE storage_column = 'low') AS low,
                       max(value) FILTER (WHERE storage_column = 'close') AS close,
                       max(value) FILTER (WHERE storage_column = 'prev_close') AS prev_close,
                       max(value) FILTER (WHERE storage_column = 'volume') AS volume,
                       max(value) FILTER (WHERE storage_column = 'amount') AS amount,
                       max(value) FILTER (WHERE storage_column = 'turnover_rate') AS turnover_rate,
                       jsonb_object_agg(metric_name, COALESCE(remark, 'null'::jsonb)) AS remark
                FROM resolved
                GROUP BY entity_id, available_at::date
            )
            INSERT INTO betalens.market_daily_fact
                (entity_id, trade_date, open, high, low, close, prev_close,
                 volume, amount, turnover_rate, remark)
            SELECT entity_id, trade_date, open, high, low, close, prev_close,
                   volume, amount, turnover_rate, remark
            FROM wide
            ON CONFLICT (entity_id, trade_date) DO UPDATE SET
                open = COALESCE(EXCLUDED.open, betalens.market_daily_fact.open),
                high = COALESCE(EXCLUDED.high, betalens.market_daily_fact.high),
                low = COALESCE(EXCLUDED.low, betalens.market_daily_fact.low),
                close = COALESCE(EXCLUDED.close, betalens.market_daily_fact.close),
                prev_close = COALESCE(EXCLUDED.prev_close, betalens.market_daily_fact.prev_close),
                volume = COALESCE(EXCLUDED.volume, betalens.market_daily_fact.volume),
                amount = COALESCE(EXCLUDED.amount, betalens.market_daily_fact.amount),
                turnover_rate = COALESCE(EXCLUDED.turnover_rate, betalens.market_daily_fact.turnover_rate),
                remark = COALESCE(betalens.market_daily_fact.remark, '{}'::jsonb)
                         || COALESCE(EXCLUDED.remark, '{}'::jsonb),
                updated_at = clock_timestamp()
        $query$, source_table, p_dataset);
    END IF;

    EXECUTE format($query$
        INSERT INTO betalens.observation_fact
            (available_at, entity_id, metric_id, period_end, value, remark)
        SELECT source.datetime::timestamp,
               entity.entity_id,
               metric.metric_id,
               CASE
                   WHEN COALESCE(source.remark->>'period_end', source.remark->>'report_date', '')
                        ~ '^\d{4}-\d{2}-\d{2}$'
                   THEN COALESCE(source.remark->>'period_end', source.remark->>'report_date')::date
                   ELSE NULL
               END,
               source.value::double precision,
               source.remark
        FROM %s source
        JOIN betalens.entity_dim entity ON entity.code = source.code
        JOIN betalens.metric_alias alias
          ON alias.logical_dataset = %L AND alias.alias = trim(source.metric)
        JOIN betalens.metric_dim metric ON metric.metric_id = alias.metric_id
        WHERE source.datetime IS NOT NULL
          AND (
              metric.storage_kind = 'observation'
              OR source.value IS NULL
              OR source.datetime::time <> metric.availability_time
          )
        ON CONFLICT (entity_id, metric_id, available_at) DO UPDATE SET
            period_end = EXCLUDED.period_end,
            value = EXCLUDED.value,
            remark = EXCLUDED.remark,
            updated_at = clock_timestamp()
    $query$, source_table, p_dataset);
END;
$$;

SELECT betalens._migrate_legacy_long('daily_market', 'stock');
SELECT betalens._migrate_legacy_long('daily_index', 'index');
SELECT betalens._migrate_legacy_long('daily_fund', 'fund');
SELECT betalens._migrate_legacy_long('daily_bond', 'bond');
SELECT betalens._migrate_legacy_long('fundamentals', 'stock');
SELECT betalens._migrate_legacy_long('macro', 'macro');
SELECT betalens._migrate_legacy_long('factors', 'stock');
DROP FUNCTION betalens._migrate_legacy_long(TEXT, TEXT);

WITH bounds AS (
    SELECT entity_id, valid_from,
           lead(valid_from) OVER (PARTITION BY entity_id ORDER BY valid_from) AS valid_to
    FROM betalens.entity_name_history
)
UPDATE betalens.entity_name_history history
SET valid_to = bounds.valid_to
FROM bounds
WHERE history.entity_id = bounds.entity_id
  AND history.valid_from = bounds.valid_from
  AND history.valid_to IS DISTINCT FROM bounds.valid_to;

DO $$
BEGIN
    IF to_regclass('public.industry') IS NOT NULL
       AND (SELECT relkind FROM pg_class WHERE oid = to_regclass('public.industry')) IN ('r', 'p') THEN
        INSERT INTO betalens.entity_dim (code, entity_type, current_name)
        SELECT DISTINCT ON (code) code, 'stock', COALESCE(NULLIF(name, ''), code)
        FROM public.industry
        WHERE code IS NOT NULL AND code <> ''
        ORDER BY code, datetime DESC
        ON CONFLICT (code) DO UPDATE
        SET current_name = EXCLUDED.current_name,
            entity_type = CASE WHEN betalens.entity_dim.entity_type = 'unknown' THEN 'stock'
                               ELSE betalens.entity_dim.entity_type END,
            updated_at = clock_timestamp();

        INSERT INTO betalens.industry_scheme_dim (scheme_name)
        SELECT DISTINCT COALESCE(NULLIF(remark->>'scheme', ''), NULLIF(metric, ''), 'unknown')
        FROM public.industry
        ON CONFLICT (scheme_name) DO NOTHING;

        INSERT INTO betalens.industry_dim (scheme_id, industry_code, industry_name)
        SELECT DISTINCT scheme.scheme_id,
               COALESCE(NULLIF(source.remark->>'ind_code', ''),
                        NULLIF(regexp_replace(source.value::text, '[.]0+$', ''), ''), 'unknown'),
               COALESCE(NULLIF(source.remark->>'ind_name', ''),
                        NULLIF(source.remark->>'industry_name', ''),
                        NULLIF(regexp_replace(source.value::text, '[.]0+$', ''), ''), 'unknown')
        FROM public.industry source
        JOIN betalens.industry_scheme_dim scheme
          ON scheme.scheme_name = COALESCE(NULLIF(source.remark->>'scheme', ''),
                                           NULLIF(source.metric, ''), 'unknown')
        ON CONFLICT (scheme_id, industry_code) DO UPDATE
        SET industry_name = EXCLUDED.industry_name;

        WITH normalized AS (
            SELECT DISTINCT ON (entity.entity_id, industry.industry_id, source.datetime)
                   entity.entity_id,
                   industry.industry_id,
                   scheme.scheme_id,
                   source.datetime::timestamp AS valid_from,
                   source.remark
            FROM public.industry source
            JOIN betalens.entity_dim entity ON entity.code = source.code
            JOIN betalens.industry_scheme_dim scheme
              ON scheme.scheme_name = COALESCE(NULLIF(source.remark->>'scheme', ''),
                                               NULLIF(source.metric, ''), 'unknown')
            JOIN betalens.industry_dim industry
              ON industry.scheme_id = scheme.scheme_id
             AND industry.industry_code = COALESCE(NULLIF(source.remark->>'ind_code', ''),
                                                   NULLIF(regexp_replace(source.value::text, '[.]0+$', ''), ''),
                                                   'unknown')
            ORDER BY entity.entity_id, industry.industry_id, source.datetime
        ), ranged AS (
            SELECT normalized.*,
                   lead(valid_from) OVER (
                       PARTITION BY entity_id, scheme_id
                       ORDER BY valid_from
                   ) AS valid_to
            FROM normalized
        )
        INSERT INTO betalens.industry_membership
            (entity_id, industry_id, valid_from, valid_to, remark)
        SELECT entity_id, industry_id, valid_from, valid_to, remark
        FROM ranged
        ON CONFLICT (entity_id, industry_id, valid_from) DO UPDATE
        SET valid_to = EXCLUDED.valid_to,
            remark = EXCLUDED.remark;
    END IF;
END;
$$;

DO $$
BEGIN
    IF to_regclass('public.index_universe') IS NOT NULL
       AND (SELECT relkind FROM pg_class WHERE oid = to_regclass('public.index_universe')) IN ('r', 'p') THEN
        IF EXISTS (
            SELECT 1 FROM public.index_universe
            WHERE remark->'constituents' IS NOT NULL
              AND jsonb_typeof(remark->'constituents') <> 'array'
        ) THEN
            RAISE EXCEPTION 'index_universe remark.constituents 必须是 JSON array';
        END IF;
        IF EXISTS (
            SELECT 1 FROM public.index_universe
            WHERE value IS NOT NULL
              AND value::bigint <> jsonb_array_length(
                  COALESCE(remark->'constituents', '[]'::jsonb)
              )
        ) THEN
            RAISE EXCEPTION 'index_universe value 与 remark.constituents 数量不一致';
        END IF;

        INSERT INTO betalens.entity_dim (code, entity_type, current_name)
        SELECT DISTINCT ON (code) code, 'index', COALESCE(NULLIF(name, ''), code)
        FROM public.index_universe
        WHERE code IS NOT NULL AND code <> ''
        ORDER BY code, datetime DESC
        ON CONFLICT (code) DO UPDATE
        SET current_name = EXCLUDED.current_name,
            entity_type = 'index',
            updated_at = clock_timestamp();

        INSERT INTO betalens.entity_dim (code, entity_type, current_name)
        SELECT DISTINCT constituent.code, 'stock', constituent.code
        FROM public.index_universe source
        CROSS JOIN LATERAL jsonb_array_elements(COALESCE(source.remark->'constituents', '[]'::jsonb))
            AS item(value)
        CROSS JOIN LATERAL (
            SELECT CASE jsonb_typeof(item.value)
                WHEN 'string' THEN item.value #>> '{}'
                WHEN 'object' THEN COALESCE(
                    item.value->>'code', item.value->>'wind_code', item.value->>'windcode'
                )
                ELSE NULL
            END AS code
        ) constituent
        WHERE constituent.code IS NOT NULL AND constituent.code <> ''
        ON CONFLICT (code) DO NOTHING;

        INSERT INTO betalens.index_snapshot
            (index_entity_id, effective_at, index_name_snapshot, remark)
        SELECT entity.entity_id,
               source.datetime::timestamp,
               COALESCE(NULLIF(source.name, ''), entity.current_name),
               source.remark - 'constituents'
        FROM public.index_universe source
        JOIN betalens.entity_dim entity ON entity.code = source.code
        ON CONFLICT (index_entity_id, effective_at) DO UPDATE
        SET index_name_snapshot = EXCLUDED.index_name_snapshot,
            remark = EXCLUDED.remark;

        INSERT INTO betalens.index_constituent
            (snapshot_id, constituent_entity_id, ordinal, weight, remark)
        SELECT snapshot.snapshot_id,
               constituent_entity.entity_id,
               item.ordinality::integer,
               CASE WHEN jsonb_typeof(item.value) = 'object'
                         AND COALESCE(item.value->>'weight', '') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
                    THEN (item.value->>'weight')::double precision ELSE NULL END,
               CASE WHEN jsonb_typeof(item.value) = 'object' THEN item.value ELSE NULL END
        FROM public.index_universe source
        JOIN betalens.entity_dim index_entity ON index_entity.code = source.code
        JOIN betalens.index_snapshot snapshot
          ON snapshot.index_entity_id = index_entity.entity_id
         AND snapshot.effective_at = source.datetime::timestamp
        CROSS JOIN LATERAL jsonb_array_elements(COALESCE(source.remark->'constituents', '[]'::jsonb))
            WITH ORDINALITY AS item(value, ordinality)
        CROSS JOIN LATERAL (
            SELECT CASE jsonb_typeof(item.value)
                WHEN 'string' THEN item.value #>> '{}'
                WHEN 'object' THEN COALESCE(item.value->>'code', item.value->>'wind_code', item.value->>'windcode')
                ELSE NULL
            END AS code
        ) constituent
        JOIN betalens.entity_dim constituent_entity ON constituent_entity.code = constituent.code
        ON CONFLICT (snapshot_id, constituent_entity_id) DO UPDATE
        SET ordinal = EXCLUDED.ordinal,
            weight = EXCLUDED.weight,
            remark = EXCLUDED.remark;
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION betalens._migrate_legacy_names(p_table TEXT)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    source_table REGCLASS;
    source_kind "char";
BEGIN
    source_table := to_regclass(format('public.%I', p_table));
    IF source_table IS NULL THEN
        RETURN;
    END IF;
    SELECT relkind INTO source_kind FROM pg_class WHERE oid = source_table;
    IF source_kind NOT IN ('r', 'p') THEN
        RETURN;
    END IF;
    EXECUTE format($query$
        WITH timeline AS (
            SELECT DISTINCT ON (entity.entity_id, source.datetime)
                   entity.entity_id,
                   source.datetime::timestamp AS valid_from,
                   source.name
            FROM %s source
            JOIN betalens.entity_dim entity ON entity.code = source.code
            WHERE source.datetime IS NOT NULL AND source.name IS NOT NULL AND source.name <> ''
            ORDER BY entity.entity_id, source.datetime, source.metric
        ), marked AS (
            SELECT timeline.*,
                   lag(name) OVER (PARTITION BY entity_id ORDER BY valid_from) AS previous_name
            FROM timeline
        )
        INSERT INTO betalens.entity_name_history (entity_id, valid_from, valid_to, name)
        SELECT entity_id, valid_from, NULL, name
        FROM marked
        WHERE previous_name IS DISTINCT FROM name
        ON CONFLICT (entity_id, valid_from) DO UPDATE SET name = EXCLUDED.name
    $query$, source_table);
END;
$$;

SELECT betalens._migrate_legacy_names('industry');
SELECT betalens._migrate_legacy_names('index_universe');
DROP FUNCTION betalens._migrate_legacy_names(TEXT);

WITH bounds AS (
    SELECT entity_id, valid_from,
           lead(valid_from) OVER (PARTITION BY entity_id ORDER BY valid_from) AS valid_to
    FROM betalens.entity_name_history
)
UPDATE betalens.entity_name_history history
SET valid_to = bounds.valid_to
FROM bounds
WHERE history.entity_id = bounds.entity_id
  AND history.valid_from = bounds.valid_from
  AND history.valid_to IS DISTINCT FROM bounds.valid_to;

DO $$
BEGIN
    IF to_regclass('public.trade_status') IS NOT NULL
       AND (SELECT relkind FROM pg_class WHERE oid = to_regclass('public.trade_status')) IN ('r', 'p') THEN
        INSERT INTO betalens.entity_dim (code, entity_type, current_name, first_trade_date)
        SELECT code,
               'stock',
               (array_agg(COALESCE(NULLIF(name, ''), code) ORDER BY datetime DESC))[1],
               min(datetime::date) FILTER (WHERE value::double precision = 1)
        FROM public.trade_status
        WHERE code IS NOT NULL AND code <> ''
        GROUP BY code
        ON CONFLICT (code) DO UPDATE
        SET current_name = EXCLUDED.current_name,
            entity_type = CASE WHEN betalens.entity_dim.entity_type = 'unknown' THEN 'stock'
                               ELSE betalens.entity_dim.entity_type END,
            first_trade_date = COALESCE(betalens.entity_dim.first_trade_date, EXCLUDED.first_trade_date),
            updated_at = clock_timestamp();

        INSERT INTO betalens.trade_status_event
            (entity_id, event_date, status, status_text, remark)
        SELECT entity.entity_id,
               source.datetime::date,
               source.value::smallint,
               COALESCE(source.remark->>'status', ''),
               source.remark
        FROM public.trade_status source
        JOIN betalens.entity_dim entity ON entity.code = source.code
        WHERE source.value::smallint <> 1
        ON CONFLICT (entity_id, event_date) DO UPDATE
        SET status = EXCLUDED.status,
            status_text = EXCLUDED.status_text,
            remark = EXCLUDED.remark;
    END IF;
END;
$$;

DO $$
BEGIN
    IF to_regclass('public.trade_status') IS NOT NULL
       AND (SELECT relkind FROM pg_class WHERE oid = to_regclass('public.trade_status')) IN ('r', 'p') THEN
        WITH timeline AS (
            SELECT DISTINCT ON (entity.entity_id, source.datetime)
                   entity.entity_id,
                   source.datetime::timestamp AS valid_from,
                   source.name
            FROM public.trade_status source
            JOIN betalens.entity_dim entity ON entity.code = source.code
            WHERE source.datetime IS NOT NULL AND source.name IS NOT NULL AND source.name <> ''
            ORDER BY entity.entity_id, source.datetime, source.metric
        ), marked AS (
            SELECT timeline.*,
                   lag(name) OVER (PARTITION BY entity_id ORDER BY valid_from) AS previous_name
            FROM timeline
        )
        INSERT INTO betalens.entity_name_history (entity_id, valid_from, valid_to, name)
        SELECT entity_id, valid_from, NULL, name
        FROM marked
        WHERE previous_name IS DISTINCT FROM name
        ON CONFLICT (entity_id, valid_from) DO UPDATE SET name = EXCLUDED.name;

        WITH bounds AS (
            SELECT entity_id, valid_from,
                   lead(valid_from) OVER (PARTITION BY entity_id ORDER BY valid_from) AS valid_to
            FROM betalens.entity_name_history
        )
        UPDATE betalens.entity_name_history history
        SET valid_to = bounds.valid_to
        FROM bounds
        WHERE history.entity_id = bounds.entity_id
          AND history.valid_from = bounds.valid_from
          AND history.valid_to IS DISTINCT FROM bounds.valid_to;
    END IF;
END;
$$;
