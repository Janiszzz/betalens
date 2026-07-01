import type { EventFilesResponse, EventStudyResult, FactorDetail, FactorProfiling, FactorSummary, RunResult, RunState, TablePage } from './types';

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

const json = async <T>(url: string, init?: RequestInit, retries = 6): Promise<T> => {
  let response: Response;
  try {
    response = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...init
    });
  } catch (err) {
    if (retries > 0 && url.startsWith('/api/')) {
      await wait(500);
      return json<T>(url, init, retries - 1);
    }
    throw err;
  }
  if (!response.ok && retries > 0 && url.startsWith('/api/') && response.status >= 500) {
    await wait(500);
    return json<T>(url, init, retries - 1);
  }
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
  clearRuns: () => json<{ cleared: number }>('/api/runs', { method: 'DELETE' }),
  run: (runId: string) => json<RunState>(`/api/runs/${runId}`),
  result: (runId: string) => json<RunResult>(`/api/runs/${runId}/result`),
  profiling: (runId: string, opts: { dateFrom?: string; dateTo?: string } = {}) => {
    const params = new URLSearchParams();
    if (opts.dateFrom) params.set('date_from', opts.dateFrom);
    if (opts.dateTo) params.set('date_to', opts.dateTo);
    const query = params.toString();
    return json<FactorProfiling>(`/api/runs/${runId}/profiling${query ? `?${query}` : ''}`);
  },
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
