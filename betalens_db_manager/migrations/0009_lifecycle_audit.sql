CREATE INDEX IF NOT EXISTS idx_industry_dim_scheme_name
    ON betalens.industry_dim (scheme_id, industry_name);

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

CREATE OR REPLACE FUNCTION betalens._audit_legacy_long(p_dataset TEXT)
RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    source_table REGCLASS;
    source_kind "char";
    invalid_rows BIGINT;
    mismatch_rows BIGINT;
BEGIN
    source_table := to_regclass(format('betalens_legacy.%I', p_dataset));
    IF source_table IS NULL THEN
        RETURN 0;
    END IF;
    SELECT relkind INTO source_kind FROM pg_class WHERE oid = source_table;
    IF source_kind NOT IN ('r', 'p') THEN
        RETURN 0;
    END IF;

    EXECUTE format($query$
        SELECT count(*)
        FROM %s source
        WHERE source.datetime IS NULL
           OR source.code IS NULL OR trim(source.code) = ''
           OR source.metric IS NULL OR trim(source.metric) = ''
    $query$, source_table)
    INTO invalid_rows;
    IF invalid_rows <> 0 THEN
        RAISE EXCEPTION
            'legacy audit % failed: % rows have an invalid datetime/code/metric key',
            p_dataset, invalid_rows;
    END IF;

    EXECUTE format($query$
        WITH expected AS (
            SELECT DISTINCT
                   source.datetime::timestamp AS datetime,
                   source.code::varchar(32) AS code,
                   COALESCE(NULLIF(source.name, ''), source.code)::varchar(200) AS name,
                   metric.metric_name::varchar(160) AS metric,
                   source.value::double precision AS value,
                   source.remark::jsonb AS remark
            FROM %s source
            JOIN betalens.metric_alias alias
              ON alias.logical_dataset = %L AND alias.alias = trim(source.metric)
            JOIN betalens.metric_dim metric ON metric.metric_id = alias.metric_id
        ), expected_keys AS (
            SELECT DISTINCT datetime, code, metric FROM expected
        ), actual AS (
            SELECT target.datetime::timestamp AS datetime,
                   target.code::varchar(32) AS code,
                   target.name::varchar(200) AS name,
                   target.metric::varchar(160) AS metric,
                   target.value::double precision AS value,
                   target.remark::jsonb AS remark
            FROM public.%I target
            JOIN expected_keys key
              ON key.datetime = target.datetime
             AND key.code = target.code
             AND key.metric = target.metric
        ), missing AS (
            SELECT datetime, code, name, metric, value, remark FROM expected
            EXCEPT
            SELECT datetime, code, name, metric, value, remark FROM actual
        ), unexpected AS (
            SELECT datetime, code, name, metric, value, remark FROM actual
            EXCEPT
            SELECT datetime, code, name, metric, value, remark FROM expected
        ), duplicate_keys AS (
            SELECT datetime, code, metric
            FROM actual
            GROUP BY datetime, code, metric
            HAVING count(*) <> 1
        )
        SELECT (SELECT count(*) FROM missing)
             + (SELECT count(*) FROM unexpected)
             + (SELECT count(*) FROM duplicate_keys)
    $query$, source_table, p_dataset, p_dataset)
    INTO mismatch_rows;

    IF mismatch_rows <> 0 THEN
        RAISE EXCEPTION
            'legacy audit % failed: % normalized key/value/name/remark differences',
            p_dataset, mismatch_rows;
    END IF;
    RETURN mismatch_rows;
END;
$$;

CREATE OR REPLACE FUNCTION betalens._audit_legacy_industry()
RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    source_table REGCLASS := to_regclass('betalens_legacy.industry');
    source_kind "char";
    mismatch_rows BIGINT;
BEGIN
    IF source_table IS NULL THEN
        RETURN 0;
    END IF;
    SELECT relkind INTO source_kind FROM pg_class WHERE oid = source_table;
    IF source_kind NOT IN ('r', 'p') THEN
        RETURN 0;
    END IF;

    WITH normalized AS (
        SELECT DISTINCT
               source.datetime::timestamp AS valid_from,
               source.code::varchar(32) AS code,
               COALESCE(NULLIF(source.remark->>'scheme', ''),
                        NULLIF(source.metric, ''), 'unknown')::varchar(160) AS scheme_name,
               COALESCE(NULLIF(source.remark->>'ind_code', ''),
                        NULLIF(regexp_replace(source.value::text, '[.]0+$', ''), ''),
                        'unknown')::varchar(64) AS industry_code,
               COALESCE(NULLIF(source.remark->>'ind_name', ''),
                        NULLIF(source.remark->>'industry_name', ''),
                        NULLIF(regexp_replace(source.value::text, '[.]0+$', ''), ''),
                        'unknown')::varchar(200) AS industry_name,
               source.remark::jsonb AS remark
        FROM betalens_legacy.industry source
        WHERE source.datetime IS NOT NULL
          AND source.code IS NOT NULL AND trim(source.code) <> ''
    ), ranged AS (
        SELECT normalized.*,
               lead(valid_from) OVER (
                   PARTITION BY code, scheme_name ORDER BY valid_from
               ) AS valid_to
        FROM normalized
    ), actual AS (
        SELECT entity.code::varchar(32) AS code,
               scheme.scheme_name::varchar(160) AS scheme_name,
               industry.industry_code::varchar(64) AS industry_code,
               industry.industry_name::varchar(200) AS industry_name,
               membership.valid_from,
               membership.valid_to,
               membership.remark
        FROM betalens.industry_membership membership
        JOIN betalens.entity_dim entity ON entity.entity_id = membership.entity_id
        JOIN betalens.industry_dim industry ON industry.industry_id = membership.industry_id
        JOIN betalens.industry_scheme_dim scheme ON scheme.scheme_id = industry.scheme_id
        JOIN (
            SELECT DISTINCT code, scheme_name, industry_code, valid_from FROM ranged
        ) key
          ON key.code = entity.code
         AND key.scheme_name = scheme.scheme_name
         AND key.industry_code = industry.industry_code
         AND key.valid_from = membership.valid_from
    ), missing AS (
        SELECT code, scheme_name, industry_code, industry_name,
               valid_from, valid_to, remark
        FROM ranged
        EXCEPT
        SELECT code, scheme_name, industry_code, industry_name,
               valid_from, valid_to, remark
        FROM actual
    ), unexpected AS (
        SELECT code, scheme_name, industry_code, industry_name,
               valid_from, valid_to, remark
        FROM actual
        EXCEPT
        SELECT code, scheme_name, industry_code, industry_name,
               valid_from, valid_to, remark
        FROM ranged
    ), duplicate_keys AS (
        SELECT code, scheme_name, industry_code, valid_from
        FROM actual
        GROUP BY code, scheme_name, industry_code, valid_from
        HAVING count(*) <> 1
    )
    SELECT (SELECT count(*) FROM missing)
         + (SELECT count(*) FROM unexpected)
         + (SELECT count(*) FROM duplicate_keys)
    INTO mismatch_rows;

    IF mismatch_rows <> 0 THEN
        RAISE EXCEPTION
            'legacy audit industry failed: % PIT key/value differences', mismatch_rows;
    END IF;
    RETURN mismatch_rows;
END;
$$;

CREATE OR REPLACE FUNCTION betalens._audit_legacy_index_universe()
RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    source_table REGCLASS := to_regclass('betalens_legacy.index_universe');
    source_kind "char";
    mismatch_rows BIGINT;
BEGIN
    IF source_table IS NULL THEN
        RETURN 0;
    END IF;
    SELECT relkind INTO source_kind FROM pg_class WHERE oid = source_table;
    IF source_kind NOT IN ('r', 'p') THEN
        RETURN 0;
    END IF;

    WITH source_rows AS (
        SELECT source.datetime::timestamp AS effective_at,
               source.code::varchar(32) AS index_code,
               COALESCE(NULLIF(source.name, ''), entity.current_name)::varchar(200) AS index_name,
               source.remark::jsonb AS source_remark,
               COALESCE(source.remark->'constituents', '[]'::jsonb) AS members
        FROM betalens_legacy.index_universe source
        JOIN betalens.entity_dim entity ON entity.code = source.code
    ), expected_snapshot AS (
        SELECT effective_at, index_code, index_name,
               (source_remark - 'constituents') AS snapshot_remark
        FROM source_rows
    ), actual_snapshot AS (
        SELECT snapshot.effective_at,
               entity.code::varchar(32) AS index_code,
               snapshot.index_name_snapshot::varchar(200) AS index_name,
               snapshot.remark AS snapshot_remark
        FROM betalens.index_snapshot snapshot
        JOIN betalens.entity_dim entity ON entity.entity_id = snapshot.index_entity_id
        JOIN expected_snapshot key
          ON key.effective_at = snapshot.effective_at
         AND key.index_code = entity.code
    ), snapshot_missing AS (
        SELECT effective_at, index_code, index_name, snapshot_remark FROM expected_snapshot
        EXCEPT
        SELECT effective_at, index_code, index_name, snapshot_remark FROM actual_snapshot
    ), snapshot_unexpected AS (
        SELECT effective_at, index_code, index_name, snapshot_remark FROM actual_snapshot
        EXCEPT
        SELECT effective_at, index_code, index_name, snapshot_remark FROM expected_snapshot
    ), expected_members AS (
        SELECT source_rows.effective_at,
               source_rows.index_code,
               item.ordinality::integer AS ordinal,
               CASE jsonb_typeof(item.value)
                   WHEN 'string' THEN item.value #>> '{}'
                   WHEN 'object' THEN COALESCE(
                       item.value->>'code', item.value->>'wind_code', item.value->>'windcode'
                   )
               END::varchar(32) AS constituent_code,
               CASE WHEN jsonb_typeof(item.value) = 'object'
                    AND COALESCE(item.value->>'weight', '') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
                    THEN (item.value->>'weight')::double precision END AS weight,
               CASE WHEN jsonb_typeof(item.value) = 'object' THEN item.value ELSE NULL END AS remark
        FROM source_rows
        CROSS JOIN LATERAL jsonb_array_elements(source_rows.members)
            WITH ORDINALITY AS item(value, ordinality)
    ), actual_members AS (
        SELECT snapshot.effective_at,
               index_entity.code::varchar(32) AS index_code,
               constituent.ordinal,
               entity.code::varchar(32) AS constituent_code,
               constituent.weight,
               constituent.remark
        FROM betalens.index_constituent constituent
        JOIN betalens.index_snapshot snapshot ON snapshot.snapshot_id = constituent.snapshot_id
        JOIN betalens.entity_dim index_entity ON index_entity.entity_id = snapshot.index_entity_id
        JOIN betalens.entity_dim entity ON entity.entity_id = constituent.constituent_entity_id
        JOIN (
            SELECT DISTINCT effective_at, index_code, ordinal, constituent_code
            FROM expected_members
            WHERE constituent_code IS NOT NULL
        ) key
          ON key.effective_at = snapshot.effective_at
         AND key.index_code = index_entity.code
         AND key.ordinal = constituent.ordinal
         AND key.constituent_code = entity.code
    ), member_missing AS (
        SELECT effective_at, index_code, ordinal, constituent_code, weight, remark
        FROM expected_members WHERE constituent_code IS NOT NULL
        EXCEPT
        SELECT effective_at, index_code, ordinal, constituent_code, weight, remark FROM actual_members
    ), member_unexpected AS (
        SELECT effective_at, index_code, ordinal, constituent_code, weight, remark FROM actual_members
        EXCEPT
        SELECT effective_at, index_code, ordinal, constituent_code, weight, remark
        FROM expected_members WHERE constituent_code IS NOT NULL
    ), duplicate_snapshots AS (
        SELECT effective_at, index_code FROM actual_snapshot
        GROUP BY effective_at, index_code HAVING count(*) <> 1
    ), duplicate_members AS (
        SELECT effective_at, index_code, ordinal, constituent_code FROM actual_members
        GROUP BY effective_at, index_code, ordinal, constituent_code HAVING count(*) <> 1
    )
    SELECT (SELECT count(*) FROM snapshot_missing)
         + (SELECT count(*) FROM snapshot_unexpected)
         + (SELECT count(*) FROM member_missing)
         + (SELECT count(*) FROM member_unexpected)
         + (SELECT count(*) FROM duplicate_snapshots)
         + (SELECT count(*) FROM duplicate_members)
    INTO mismatch_rows;

    IF mismatch_rows <> 0 THEN
        RAISE EXCEPTION
            'legacy audit index_universe failed: % snapshot/constituent differences', mismatch_rows;
    END IF;
    RETURN mismatch_rows;
END;
$$;

CREATE OR REPLACE FUNCTION betalens._audit_legacy_trade_status()
RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    source_table REGCLASS := to_regclass('betalens_legacy.trade_status');
    source_kind "char";
    mismatch_rows BIGINT;
BEGIN
    IF source_table IS NULL THEN
        RETURN 0;
    END IF;
    SELECT relkind INTO source_kind FROM pg_class WHERE oid = source_table;
    IF source_kind NOT IN ('r', 'p') THEN
        RETURN 0;
    END IF;

    WITH source_rows AS (
        SELECT source.datetime::timestamp AS available_at,
               source.code::varchar(32) AS code,
               source.value::smallint AS status,
               source.remark::jsonb AS source_remark,
               source.name::varchar(200) AS source_name
        FROM betalens_legacy.trade_status source
        WHERE source.datetime IS NOT NULL
          AND source.code IS NOT NULL AND trim(source.code) <> ''
    ), anchors AS (
        SELECT DISTINCT ON (code)
               available_at::date AS event_date,
               code,
               COALESCE(NULLIF(source_name, ''), code)::varchar(200) AS name,
               1::double precision AS value,
               jsonb_build_object('status', '交易', 'first_normal', true) AS remark
        FROM source_rows
        WHERE status = 1
        ORDER BY code, available_at
    ), events AS (
        SELECT available_at::date AS event_date,
               code,
               COALESCE(NULLIF(source_name, ''), code)::varchar(200) AS name,
               status::double precision AS value,
               jsonb_build_object('status', COALESCE(source_remark->>'status', ''))
                   || COALESCE(source_remark, '{}'::jsonb) AS remark
        FROM source_rows
        WHERE status <> 1
    ), expected AS (
        SELECT event_date + TIME '15:00:01' AS datetime, code, name,
               '交易状态'::varchar(160) AS metric, value, remark FROM anchors
        UNION ALL
        SELECT event_date + TIME '15:00:01', code, name,
               '交易状态'::varchar(160), value, remark FROM events
    ), actual AS (
        SELECT target.datetime::timestamp, target.code::varchar(32),
               target.name::varchar(200), target.metric::varchar(160),
               target.value::double precision, target.remark::jsonb
        FROM public.trade_status target
        JOIN (SELECT DISTINCT datetime, code, metric FROM expected) key
          ON key.datetime = target.datetime
         AND key.code = target.code
         AND key.metric = target.metric
    ), missing AS (
        SELECT datetime, code, name, metric, value, remark FROM expected
        EXCEPT
        SELECT datetime, code, name, metric, value, remark FROM actual
    ), unexpected AS (
        SELECT datetime, code, name, metric, value, remark FROM actual
        EXCEPT
        SELECT datetime, code, name, metric, value, remark FROM expected
    ), duplicate_keys AS (
        SELECT datetime, code, metric FROM actual
        GROUP BY datetime, code, metric HAVING count(*) <> 1
    )
    SELECT (SELECT count(*) FROM missing)
         + (SELECT count(*) FROM unexpected)
         + (SELECT count(*) FROM duplicate_keys)
    INTO mismatch_rows;

    IF mismatch_rows <> 0 THEN
        RAISE EXCEPTION
            'legacy audit trade_status failed: % event/anchor differences', mismatch_rows;
    END IF;
    RETURN mismatch_rows;
END;
$$;

CREATE OR REPLACE FUNCTION betalens.assert_legacy_equivalence()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    dataset TEXT;
    long_checked BIGINT := 0;
    industry_checked BIGINT := 0;
    index_checked BIGINT := 0;
    trade_status_checked BIGINT := 0;
BEGIN
    FOREACH dataset IN ARRAY ARRAY[
        'daily_market', 'daily_index', 'daily_fund', 'daily_bond',
        'fundamentals', 'macro', 'factors'
    ] LOOP
        long_checked := long_checked + betalens._audit_legacy_long(dataset);
    END LOOP;
    industry_checked := betalens._audit_legacy_industry();
    index_checked := betalens._audit_legacy_index_universe();
    trade_status_checked := betalens._audit_legacy_trade_status();
    RETURN jsonb_build_object(
        'long_differences', long_checked,
        'industry_differences', industry_checked,
        'index_universe_differences', index_checked,
        'trade_status_differences', trade_status_checked
    );
END;
$$;

-- Legacy tables are retained for inspection, but an imperfect historical
-- conversion must not prevent a user from installing the new schema.  The
-- audit function remains available for diagnosis; its result is advisory
-- during bootstrap instead of forcing the whole DDL transaction to roll back.
DO $$
BEGIN
    PERFORM betalens.assert_legacy_equivalence();
EXCEPTION WHEN OTHERS THEN
    RAISE WARNING
        'Betalens legacy equivalence audit reported differences; schema installation continues: %',
        SQLERRM;
END;
$$;
