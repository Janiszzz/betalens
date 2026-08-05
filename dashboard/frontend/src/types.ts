export type StrategyType = 'cross_sectional' | 'timing';

export type FactorSummary = {
  factor_class: string;
  name: string;
  strategy_type: StrategyType;
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
  group?: string;
};

export type TimingPayload = {
  metrics: Metric[];
  charts: {
    navPrice: Array<Record<string, number | string | null>>;
    tradeMarkers: Array<Record<string, number | string | null>>;
    position: Array<Record<string, number | string | null>>;
    drawdown: Array<Record<string, number | string | null>>;
    dailyPnl: Array<Record<string, number | string | null>>;
    tradeReturns: Array<Record<string, number | string | null>>;
    predictionScatter: Array<Record<string, number | string | null>>;
    openForwardReturns: Array<Record<string, number | string | null>>;
  };
  tables: {
    tradeSegments: Array<Record<string, unknown>>;
    prediction: Array<Record<string, unknown>>;
  };
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
  timing: TimingPayload;
  charts: {
    nav: Array<Record<string, number | string>>;
    drawdown: Array<Record<string, number | string>>;
    dailyPnl: Array<Record<string, number | string>>;
    dailyAmount: Array<Record<string, number | string>>;
    positionWeight: Array<Record<string, number | string>>;
    rebalanceHoldings: Array<Record<string, number | string | null>>;
    groupNav: Array<Record<string, number | string | null>>;
    tradePairs: Array<Record<string, number | string | null>>;
    annualTrade: Array<Record<string, number | string | null>>;
    profiling: FactorProfiling;
  };
  tables: {
    trades: TableMeta;
    positions: TableMeta;
  };
  diagnostics?: {
    pitValidation: Array<Record<string, unknown>>;
    neutralizeStats: Array<Record<string, unknown>>;
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

export type FactorProfiling = {
  available: boolean;
  dateFrom?: string | null;
  dateTo?: string | null;
  summary: Record<string, number | string | null>;
  histogram: Array<Record<string, number | string | null>>;
  ecdf: Array<Record<string, number | string | null>>;
  quantiles: Array<Record<string, number | string | null>>;
  tests: Array<Record<string, number | string | null>>;
  timeseries: Array<Record<string, number | string | null>>;
  autocorrelation: Array<Record<string, number | string | null>>;
  turnover: Array<Record<string, number | string | null>>;
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

export type EventStudyComparison = {
  mode: 'compare';
  events: Array<{
    eventId: number;
    eventDate: string | null;
  }>;
  validCodes: string[];
  skippedCodes: Array<{
    code: string;
    reason: string;
  }>;
  totalEventCount: number;
  displayedEventCount: number;
  truncated: boolean;
  summaryByCode: Array<Record<string, number | string | null>>;
  dailyByCode: Array<Record<string, number | string | null>>;
  cumulativeByCode: Array<Record<string, number | string | null>>;
  eventCumulativeByCode: Array<Record<string, number | string | null>>;
};

export type EventStudyAsset = {
  code: string;
  name: string | null;
  label: string;
};

export type EventStudyResult = {
  eventFile: {
    id: string;
    name: string;
    path: string;
  };
  assets?: EventStudyAsset[];
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
  comparison?: EventStudyComparison;
};
