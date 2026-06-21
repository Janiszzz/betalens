import type { EventFilesResponse, EventStudyResult, FactorDetail, FactorSummary, RunResult, RunState, TablePage } from './types';

const json = async <T>(url: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...init
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // keep default
    }
    throw new Error(detail);
  }
  return response.json();
};

export const api = {
  factors: () => json<FactorSummary[]>('/api/factors'),
  eventFiles: () => json<EventFilesResponse>('/api/eventstudy/files'),
  runEventStudy: (body: Record<string, unknown>) =>
    json<EventStudyResult>('/api/eventstudy/run', { method: 'POST', body: JSON.stringify(body) }),
  factor: (factorClass: string, name: string) =>
    json<FactorDetail>(`/api/factors/${encodeURIComponent(factorClass)}/${encodeURIComponent(name)}`),
  startRun: (body: { factor_class: string; name: string; parameters: Record<string, unknown>; compute_kwargs: Record<string, unknown> }) =>
    json<{ run_id: string }>('/api/runs', { method: 'POST', body: JSON.stringify(body) }),
  run: (runId: string) => json<RunState>(`/api/runs/${runId}`),
  result: (runId: string) => json<RunResult>(`/api/runs/${runId}/result`),
  table: (
    runId: string,
    kind: 'trades' | 'positions',
    opts: { page?: number; size?: number; query?: string; filters?: Record<string, string>; dateFrom?: string; dateTo?: string } = {}
  ) => {
    const params = new URLSearchParams();
    params.set('page', String(opts.page ?? 1));
    params.set('size', String(opts.size ?? 50));
    if (opts.query) params.set('query', opts.query);
    if (opts.dateFrom) params.set('date_from', opts.dateFrom);
    if (opts.dateTo) params.set('date_to', opts.dateTo);
    for (const [col, val] of Object.entries(opts.filters ?? {})) {
      if (val) params.set(`filter.${col}`, val);
    }
    return json<TablePage>(`/api/runs/${runId}/table/${kind}?${params.toString()}`);
  },
  downloadUrl: (runId: string, kind: string) => `/api/runs/${runId}/download/${kind}`
};
