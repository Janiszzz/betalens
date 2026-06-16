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
    positionValue: Array<Record<string, number | string>>;
  };
  trades: Array<Record<string, unknown>>;
  positions: Array<Record<string, unknown>>;
  downloads: Record<string, { path: string | null; exists: boolean }>;
};
