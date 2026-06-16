import type { FactorDetail, FactorSummary, RunResult, RunState } from './types';

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
  factor: (factorClass: string, name: string) =>
    json<FactorDetail>(`/api/factors/${encodeURIComponent(factorClass)}/${encodeURIComponent(name)}`),
  startRun: (body: { factor_class: string; name: string; parameters: Record<string, unknown>; compute_kwargs: Record<string, unknown> }) =>
    json<{ run_id: string }>('/api/runs', { method: 'POST', body: JSON.stringify(body) }),
  run: (runId: string) => json<RunState>(`/api/runs/${runId}`),
  result: (runId: string) => json<RunResult>(`/api/runs/${runId}/result`),
  downloadUrl: (runId: string, kind: string) => `/api/runs/${runId}/download/${kind}`
};
