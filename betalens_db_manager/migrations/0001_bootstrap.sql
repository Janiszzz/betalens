CREATE SCHEMA IF NOT EXISTS betalens;

CREATE TABLE IF NOT EXISTS betalens.schema_migration (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum CHAR(64) NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    execution_ms INTEGER NOT NULL DEFAULT 0 CHECK (execution_ms >= 0)
);

