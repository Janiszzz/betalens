CREATE TABLE betalens.observation_fact (
    available_at TIMESTAMP NOT NULL,
    entity_id BIGINT NOT NULL REFERENCES betalens.entity_dim(entity_id) ON DELETE CASCADE,
    metric_id INTEGER NOT NULL REFERENCES betalens.metric_dim(metric_id) ON DELETE RESTRICT,
    period_end DATE,
    value DOUBLE PRECISION,
    remark JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (entity_id, metric_id, available_at)
) PARTITION BY RANGE (available_at);

CREATE INDEX idx_observation_metric_time_entity
    ON betalens.observation_fact (metric_id, available_at, entity_id);
