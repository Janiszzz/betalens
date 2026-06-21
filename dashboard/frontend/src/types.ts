export type FactorSummary = {
  factor_class: string;
  name: string;
  formula: string;
  logic: string;
  source: string;
  inputs: Record<string, string>;
  defaults: Record<string, unknown>;
};

export type FactorDetail = FactorSummary & {
  doc: string;
  compute_kwargs: Record<string, unknown>;
  script_path: string;
  factor_dir: string;
};

export type RunStatus = 'queued' | 'running' | 'completed' | 'failed';

export type RunState = {
  run_id: string;
  status: RunStatus;
  factor_class: string;
  name: string;
  started_at: string | null;
  finished_at: string | null;
  elapsed_seconds: number;
  error: string | null;
  log_size: number;
};

export type Metric = {
  label: string;
  value: number | string | null;
  format: 'number' | 'percent';
};

export type RunResult = {
  run: RunState;
  factor: {
    class: string;
    name: string;
    parameters: Record<string, unknown>;
    compute_kwargs: Record<string, unknown>;
  };
  metrics: Metric[];
  charts: {
    nav: Array<Record<string, number | string>>;
    drawdown: Array<Record<string, number | string>>;
    dailyPnl: Array<Record<string, number | string>>;
    dailyAmount: Array<Record<string, number | string>>;
    positionWeight: Array<Record<string, number | string>>;
    rebalanceHoldings: Array<Record<string, number | string | null>>;
  };
  tables: {
    trades: TableMeta;
    positions: TableMeta;
  };
  downloads: Record<string, { path: string | null; exists: boolean }>;
};

export type TableMeta = {
  total: number;
  columns: string[];
};

export type TablePage = {
  rows: Array<Record<string, unknown>>;
  total: number;
  page: number;
  size: number;
  pages: number;
};

export type EventFile = {
  id: string;
  name: string;
  path: string;
  eventCount: number;
  dateFrom: string;
  dateTo: string;
  columns: string[];
  sample: Array<Record<string, unknown>>;
  error?: string;
};

export type EventFilesResponse = {
  defaults: Record<string, unknown>;
  files: EventFile[];
};

export type EventStudyResult = {
  eventFile: {
    id: string;
    name: string;
    path: string;
  };
  parameters: Record<string, unknown>;
  summary: Record<string, number | string | string[] | null>;
  charts: {
    dailyStats: Array<Record<string, number | string | null>>;
    cumulativeStats: Array<Record<string, number | string | null>>;
    returnsMatrix: Array<Record<string, number | string | null>>;
    cumulativeReturnsMatrix: Array<Record<string, number | string | null>>;
  };
  tables: {
    dailyStats: Array<Record<string, unknown>>;
    cumulativeStats: Array<Record<string, unknown>>;
    events: Array<Record<string, unknown>>;
  };
};
