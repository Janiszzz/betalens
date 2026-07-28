CREATE SCHEMA IF NOT EXISTS betalens_legacy;

CREATE OR REPLACE FUNCTION betalens._assert_legacy_long_migrated(
    p_dataset TEXT,
    p_entity_type TEXT
) RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    source_table REGCLASS;
    source_kind "char";
    expected_rows BIGINT;
    actual_rows BIGINT;
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
        SELECT count(*)
        FROM (
            SELECT DISTINCT source.datetime::timestamp, source.code, alias.metric_id
            FROM %s source
            JOIN betalens.metric_alias alias
              ON alias.logical_dataset = %L AND alias.alias = trim(source.metric)
            WHERE source.datetime IS NOT NULL AND source.code IS NOT NULL
        ) expected
    $query$, source_table, p_dataset)
    INTO expected_rows;

    SELECT count(*) INTO actual_rows
    FROM (
        SELECT fact.trade_date, entity.code, metric.metric_id
        FROM betalens.market_daily_fact fact
        JOIN betalens.entity_dim entity
          ON entity.entity_id = fact.entity_id AND entity.entity_type = p_entity_type
        JOIN betalens.metric_dim metric
          ON metric.logical_dataset = p_dataset AND metric.storage_kind = 'core'
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
        SELECT observation.available_at::date, entity.code, metric.metric_id
        FROM betalens.observation_fact observation
        JOIN betalens.entity_dim entity ON entity.entity_id = observation.entity_id
        JOIN betalens.metric_dim metric ON metric.metric_id = observation.metric_id
        WHERE metric.logical_dataset = p_dataset
    ) migrated;

    IF actual_rows < expected_rows THEN
        RAISE EXCEPTION
            '拒绝切换旧表 public.%：期望至少 % 个规范化 key，实际仅迁移 % 个',
            p_dataset, expected_rows, actual_rows;
    END IF;
END;
$$;

SELECT betalens._assert_legacy_long_migrated('daily_market', 'stock');
SELECT betalens._assert_legacy_long_migrated('daily_index', 'index');
SELECT betalens._assert_legacy_long_migrated('daily_fund', 'fund');
SELECT betalens._assert_legacy_long_migrated('daily_bond', 'bond');
SELECT betalens._assert_legacy_long_migrated('fundamentals', 'stock');
SELECT betalens._assert_legacy_long_migrated('macro', 'macro');
SELECT betalens._assert_legacy_long_migrated('factors', 'stock');
DROP FUNCTION betalens._assert_legacy_long_migrated(TEXT, TEXT);

DO $$
DECLARE
    expected_rows BIGINT;
    actual_rows BIGINT;
BEGIN
    IF to_regclass('public.industry') IS NOT NULL
       AND (SELECT relkind FROM pg_class WHERE oid = to_regclass('public.industry')) IN ('r', 'p') THEN
        SELECT count(*) INTO expected_rows
        FROM (
            SELECT DISTINCT code,
                   COALESCE(NULLIF(remark->>'scheme', ''), NULLIF(metric, ''), 'unknown'),
                   COALESCE(NULLIF(remark->>'ind_code', ''),
                            NULLIF(regexp_replace(value::text, '[.]0+$', ''), ''), 'unknown'),
                   datetime::timestamp
            FROM public.industry
        ) expected;
        SELECT count(*) INTO actual_rows FROM betalens.industry_membership;
        IF actual_rows < expected_rows THEN
            RAISE EXCEPTION '拒绝切换 industry：期望 % 条 membership，实际 % 条',
                expected_rows, actual_rows;
        END IF;
    END IF;

    IF to_regclass('public.index_universe') IS NOT NULL
       AND (SELECT relkind FROM pg_class WHERE oid = to_regclass('public.index_universe')) IN ('r', 'p') THEN
        SELECT count(*) INTO expected_rows
        FROM (SELECT DISTINCT code, datetime::timestamp FROM public.index_universe) expected;
        SELECT count(*) INTO actual_rows FROM betalens.index_snapshot;
        IF actual_rows < expected_rows THEN
            RAISE EXCEPTION '拒绝切换 index_universe：期望 % 个 snapshot，实际 % 个',
                expected_rows, actual_rows;
        END IF;
        SELECT count(*) INTO expected_rows
        FROM (
            SELECT DISTINCT source.code, source.datetime::timestamp,
                   CASE jsonb_typeof(item.value)
                       WHEN 'string' THEN item.value #>> '{}'
                       WHEN 'object' THEN COALESCE(
                           item.value->>'code', item.value->>'wind_code', item.value->>'windcode'
                       )
                   END AS constituent_code
            FROM public.index_universe source
            CROSS JOIN LATERAL jsonb_array_elements(
                COALESCE(source.remark->'constituents', '[]'::jsonb)
            ) AS item(value)
        ) expected
        WHERE constituent_code IS NOT NULL;
        SELECT count(*) INTO actual_rows FROM betalens.index_constituent;
        IF actual_rows < expected_rows THEN
            RAISE EXCEPTION '拒绝切换 index_universe：期望 % 个 constituent，实际 % 个',
                expected_rows, actual_rows;
        END IF;
    END IF;

    IF to_regclass('public.trade_status') IS NOT NULL
       AND (SELECT relkind FROM pg_class WHERE oid = to_regclass('public.trade_status')) IN ('r', 'p') THEN
        SELECT count(*) INTO expected_rows
        FROM (
            SELECT DISTINCT code, datetime::date
            FROM public.trade_status
            WHERE value::smallint <> 1
        ) expected;
        SELECT count(*) INTO actual_rows FROM betalens.trade_status_event;
        IF actual_rows < expected_rows THEN
            RAISE EXCEPTION '拒绝切换 trade_status：期望 % 个异常事件，实际 % 个',
                expected_rows, actual_rows;
        END IF;
        IF EXISTS (
            SELECT 1
            FROM public.trade_status source
            WHERE source.value::smallint = 1
              AND NOT EXISTS (
                  SELECT 1 FROM betalens.entity_dim entity
                  WHERE entity.code = source.code AND entity.first_trade_date IS NOT NULL
              )
        ) THEN
            RAISE EXCEPTION '拒绝切换 trade_status：存在未迁移的首次正常交易锚点';
        END IF;
    END IF;
END;
$$;

DO $$
DECLARE
    relation_name TEXT;
    source_relation REGCLASS;
    relation_kind "char";
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'daily_market', 'daily_index', 'daily_fund', 'daily_bond',
        'fundamentals', 'macro', 'factors',
        'industry', 'index_universe', 'trade_status'
    ] LOOP
        source_relation := to_regclass(format('public.%I', relation_name));
        IF source_relation IS NULL THEN
            CONTINUE;
        END IF;
        SELECT relkind INTO relation_kind FROM pg_class WHERE oid = source_relation;
        IF relation_kind NOT IN ('r', 'p') THEN
            CONTINUE;
        END IF;
        IF to_regclass(format('betalens_legacy.%I', relation_name)) IS NOT NULL THEN
            RAISE EXCEPTION '无法保护旧表 public.%：betalens_legacy.% 已存在',
                relation_name, relation_name;
        END IF;
        EXECUTE format('ALTER TABLE public.%I SET SCHEMA betalens_legacy', relation_name);
    END LOOP;
END;
$$;

CREATE OR REPLACE FUNCTION betalens.entity_name_at(
    p_entity_id BIGINT,
    p_available_at TIMESTAMP
) RETURNS VARCHAR(200)
LANGUAGE SQL
STABLE
PARALLEL SAFE
AS $$
    SELECT COALESCE(
        (
            SELECT history.name
            FROM betalens.entity_name_history history
            WHERE history.entity_id = p_entity_id
              AND history.valid_from <= p_available_at
              AND (history.valid_to IS NULL OR history.valid_to > p_available_at)
            ORDER BY history.valid_from DESC
            LIMIT 1
        ),
        (
            SELECT entity.current_name
            FROM betalens.entity_dim entity
            WHERE entity.entity_id = p_entity_id
        )
    )::varchar(200)
$$;

CREATE OR REPLACE VIEW public.daily_market AS
SELECT fact.trade_date + metric.availability_time AS datetime,
       entity.code,
       betalens.entity_name_at(
           entity.entity_id, fact.trade_date + metric.availability_time
       ) AS name,
       metric.metric_name AS metric,
       value.value,
       NULLIF(fact.remark->metric.metric_name, 'null'::jsonb) AS remark
FROM betalens.market_daily_fact fact
JOIN betalens.entity_dim entity
  ON entity.entity_id = fact.entity_id AND entity.entity_type = 'stock'
JOIN betalens.metric_dim metric
  ON metric.logical_dataset = 'daily_market' AND metric.storage_kind = 'core'
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
    END::double precision AS value
) value
WHERE value.value IS NOT NULL
UNION ALL
SELECT observation.available_at,
       entity.code,
       betalens.entity_name_at(entity.entity_id, observation.available_at),
       metric.metric_name,
       observation.value,
       observation.remark
FROM betalens.observation_fact observation
JOIN betalens.entity_dim entity ON entity.entity_id = observation.entity_id
JOIN betalens.metric_dim metric ON metric.metric_id = observation.metric_id
WHERE metric.logical_dataset = 'daily_market';

CREATE OR REPLACE VIEW public.daily_index AS
SELECT fact.trade_date + metric.availability_time AS datetime,
       entity.code,
       betalens.entity_name_at(
           entity.entity_id, fact.trade_date + metric.availability_time
       ) AS name,
       metric.metric_name AS metric,
       value.value,
       NULLIF(fact.remark->metric.metric_name, 'null'::jsonb) AS remark
FROM betalens.market_daily_fact fact
JOIN betalens.entity_dim entity
  ON entity.entity_id = fact.entity_id AND entity.entity_type = 'index'
JOIN betalens.metric_dim metric
  ON metric.logical_dataset = 'daily_index' AND metric.storage_kind = 'core'
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
    END::double precision AS value
) value
WHERE value.value IS NOT NULL
UNION ALL
SELECT observation.available_at,
       entity.code,
       betalens.entity_name_at(entity.entity_id, observation.available_at),
       metric.metric_name,
       observation.value,
       observation.remark
FROM betalens.observation_fact observation
JOIN betalens.entity_dim entity ON entity.entity_id = observation.entity_id
JOIN betalens.metric_dim metric ON metric.metric_id = observation.metric_id
WHERE metric.logical_dataset = 'daily_index';

CREATE OR REPLACE VIEW public.daily_fund AS
SELECT fact.trade_date + metric.availability_time AS datetime,
       entity.code,
       betalens.entity_name_at(
           entity.entity_id, fact.trade_date + metric.availability_time
       ) AS name,
       metric.metric_name AS metric,
       value.value,
       NULLIF(fact.remark->metric.metric_name, 'null'::jsonb) AS remark
FROM betalens.market_daily_fact fact
JOIN betalens.entity_dim entity
  ON entity.entity_id = fact.entity_id AND entity.entity_type = 'fund'
JOIN betalens.metric_dim metric
  ON metric.logical_dataset = 'daily_fund' AND metric.storage_kind = 'core'
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
    END::double precision AS value
) value
WHERE value.value IS NOT NULL
UNION ALL
SELECT observation.available_at,
       entity.code,
       betalens.entity_name_at(entity.entity_id, observation.available_at),
       metric.metric_name,
       observation.value,
       observation.remark
FROM betalens.observation_fact observation
JOIN betalens.entity_dim entity ON entity.entity_id = observation.entity_id
JOIN betalens.metric_dim metric ON metric.metric_id = observation.metric_id
WHERE metric.logical_dataset = 'daily_fund';

CREATE OR REPLACE VIEW public.daily_bond AS
SELECT fact.trade_date + metric.availability_time AS datetime,
       entity.code,
       betalens.entity_name_at(
           entity.entity_id, fact.trade_date + metric.availability_time
       ) AS name,
       metric.metric_name AS metric,
       value.value,
       NULLIF(fact.remark->metric.metric_name, 'null'::jsonb) AS remark
FROM betalens.market_daily_fact fact
JOIN betalens.entity_dim entity
  ON entity.entity_id = fact.entity_id AND entity.entity_type = 'bond'
JOIN betalens.metric_dim metric
  ON metric.logical_dataset = 'daily_bond' AND metric.storage_kind = 'core'
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
    END::double precision AS value
) value
WHERE value.value IS NOT NULL
UNION ALL
SELECT observation.available_at,
       entity.code,
       betalens.entity_name_at(entity.entity_id, observation.available_at),
       metric.metric_name,
       observation.value,
       observation.remark
FROM betalens.observation_fact observation
JOIN betalens.entity_dim entity ON entity.entity_id = observation.entity_id
JOIN betalens.metric_dim metric ON metric.metric_id = observation.metric_id
WHERE metric.logical_dataset = 'daily_bond';

CREATE OR REPLACE VIEW public.fundamentals AS
SELECT observation.available_at AS datetime,
       entity.code,
       betalens.entity_name_at(entity.entity_id, observation.available_at) AS name,
       metric.metric_name AS metric,
       observation.value,
       observation.remark
FROM betalens.observation_fact observation
JOIN betalens.entity_dim entity ON entity.entity_id = observation.entity_id
JOIN betalens.metric_dim metric ON metric.metric_id = observation.metric_id
WHERE metric.logical_dataset = 'fundamentals';

CREATE OR REPLACE VIEW public.macro AS
SELECT observation.available_at AS datetime,
       entity.code,
       betalens.entity_name_at(entity.entity_id, observation.available_at) AS name,
       metric.metric_name AS metric,
       observation.value,
       observation.remark
FROM betalens.observation_fact observation
JOIN betalens.entity_dim entity ON entity.entity_id = observation.entity_id
JOIN betalens.metric_dim metric ON metric.metric_id = observation.metric_id
WHERE metric.logical_dataset = 'macro';

CREATE OR REPLACE VIEW public.factors AS
SELECT observation.available_at AS datetime,
       entity.code,
       betalens.entity_name_at(entity.entity_id, observation.available_at) AS name,
       metric.metric_name AS metric,
       observation.value,
       observation.remark
FROM betalens.observation_fact observation
JOIN betalens.entity_dim entity ON entity.entity_id = observation.entity_id
JOIN betalens.metric_dim metric ON metric.metric_id = observation.metric_id
WHERE metric.logical_dataset = 'factors';

CREATE OR REPLACE VIEW public.industry AS
SELECT membership.valid_from AS datetime,
       entity.code,
       betalens.entity_name_at(entity.entity_id, membership.valid_from) AS name,
       scheme.scheme_name AS metric,
       NULLIF(
           regexp_replace(industry.industry_code, '[^0-9]', '', 'g'),
           ''
       )::double precision AS value,
       jsonb_build_object(
           'ind_name', industry.industry_name,
           'ind_code', industry.industry_code,
           'scheme', scheme.scheme_name
       ) || COALESCE(membership.remark, '{}'::jsonb) AS remark
FROM betalens.industry_membership membership
JOIN betalens.entity_dim entity ON entity.entity_id = membership.entity_id
JOIN betalens.industry_dim industry ON industry.industry_id = membership.industry_id
JOIN betalens.industry_scheme_dim scheme ON scheme.scheme_id = industry.scheme_id;

CREATE OR REPLACE VIEW public.index_universe AS
SELECT snapshot.effective_at AS datetime,
       index_entity.code,
       COALESCE(NULLIF(snapshot.index_name_snapshot, ''), index_entity.current_name)::varchar(200) AS name,
       'universe'::varchar(160) AS metric,
       count(constituent.constituent_entity_id)::double precision AS value,
       jsonb_build_object(
           'index_code', index_entity.code,
           'index_name', COALESCE(NULLIF(snapshot.index_name_snapshot, ''), index_entity.current_name),
           'constituents', COALESCE(
               jsonb_agg(constituent_entity.code ORDER BY constituent.ordinal, constituent_entity.code)
                   FILTER (WHERE constituent.constituent_entity_id IS NOT NULL),
               '[]'::jsonb
           )
       ) || COALESCE(snapshot.remark, '{}'::jsonb) AS remark
FROM betalens.index_snapshot snapshot
JOIN betalens.entity_dim index_entity ON index_entity.entity_id = snapshot.index_entity_id
LEFT JOIN betalens.index_constituent constituent ON constituent.snapshot_id = snapshot.snapshot_id
LEFT JOIN betalens.entity_dim constituent_entity
  ON constituent_entity.entity_id = constituent.constituent_entity_id
GROUP BY snapshot.snapshot_id, snapshot.effective_at, index_entity.code,
         index_entity.current_name, snapshot.index_name_snapshot, snapshot.remark;

CREATE OR REPLACE VIEW public.trade_status AS
SELECT entity.first_trade_date + TIME '15:00:01' AS datetime,
       entity.code,
       betalens.entity_name_at(
           entity.entity_id, entity.first_trade_date + TIME '15:00:01'
       ) AS name,
       '交易状态'::varchar(160) AS metric,
       1::double precision AS value,
       jsonb_build_object('status', '交易', 'first_normal', true) AS remark
FROM betalens.entity_dim entity
WHERE entity.first_trade_date IS NOT NULL
UNION ALL
SELECT event.event_date + TIME '15:00:01',
       entity.code,
       betalens.entity_name_at(
           entity.entity_id, event.event_date + TIME '15:00:01'
       ),
       '交易状态'::varchar(160),
       event.status::double precision,
       jsonb_build_object('status', event.status_text) || COALESCE(event.remark, '{}'::jsonb)
FROM betalens.trade_status_event event
JOIN betalens.entity_dim entity ON entity.entity_id = event.entity_id;
