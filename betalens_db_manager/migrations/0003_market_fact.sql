CREATE TABLE betalens.market_daily_fact (
    entity_id BIGINT NOT NULL REFERENCES betalens.entity_dim(entity_id) ON DELETE CASCADE,
    trade_date DATE NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    prev_close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    amount DOUBLE PRECISION,
    turnover_rate DOUBLE PRECISION,
    remark JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (entity_id, trade_date)
);
CREATE INDEX idx_market_daily_fact_trade_date_entity
    ON betalens.market_daily_fact (trade_date, entity_id);

