CREATE TABLE betalens.trade_calendar_day (
    exchange VARCHAR(32) NOT NULL,
    trade_date DATE NOT NULL,
    PRIMARY KEY (exchange, trade_date),
    CHECK (btrim(exchange) <> '')
);

INSERT INTO betalens.dataset_coverage (logical_dataset)
VALUES ('trade_calendar')
ON CONFLICT (logical_dataset) DO NOTHING;

CREATE OR REPLACE VIEW public.trade_calendar AS
SELECT calendar.trade_date::timestamp AS datetime,
       calendar.exchange AS code,
       calendar.exchange AS name,
       '交易日'::varchar(160) AS metric,
       1::double precision AS value,
       NULL::jsonb AS remark
FROM betalens.trade_calendar_day calendar
GROUP BY calendar.exchange, calendar.trade_date;

COMMENT ON TABLE betalens.trade_calendar_day IS
    'Exchange trading calendars, one continuous date sequence per exchange.';
COMMENT ON COLUMN betalens.trade_calendar_day.exchange IS
    'Normalized exchange/calendar code, for example SHSE or NIB.';
COMMENT ON COLUMN betalens.trade_calendar_day.trade_date IS
    'A trading date in the exchange calendar.';
