import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { ArrowDown, ArrowDownToLine, ArrowLeft, ArrowRight, ArrowUp, ArrowUpDown, ChevronRight, Columns3, Search } from 'lucide-react';
import { BacktestResultView, ResultNavigation } from '../../dashboard/frontend/src/App';
import type { ResultTab } from '../../dashboard/frontend/src/App';
import type { BacktestResultClient } from '../../dashboard/frontend/src/result-client';
import type { RunResult } from '../../dashboard/frontend/src/types';
import dashboardStyles from '../../dashboard/frontend/src/styles.css?inline';
import { getJson, url, useApi } from './http';

type Dict = Record<string, any>;
type Metric = { key: string; label: string; format: string };
type Catalog = { items: Dict[]; total: number; page: number; page_size: number;
  metrics: Metric[]; frequencies: string[]; universes: string[] };
const initialColumns = ['ic', 'rank_ic', 'ic_ir', 'annual_return', 'sharpe', 'max_drawdown', 'turnover'];
const timestamp = (value: string) => new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Shanghai', dateStyle: 'short', timeStyle: 'short' }).format(new Date(value));
const valueText = (value: unknown, format = 'number') => typeof value === 'number'
  ? format === 'percent' ? `${(value * 100).toFixed(2)}%` : value.toFixed(4) : '—';
const range = (period?: Dict) => period?.start || period?.end ? `${period.start || '—'} ~ ${period.end || '—'}` : '—';
const paramsText = (params: Dict) => Object.entries(params).filter(([key]) => key !== 'factor_id')
  .map(([key, value]) => `${key}=${typeof value === 'object' ? JSON.stringify(value) : String(value)}`).join(' · ');

export function FactorCatalog({ factors, revision, onChoose }: {
  factors: Dict[]; revision: number; onChoose: (factor: any, experimentId?: string) => void;
}) {
  const [search, setSearch] = useState('');
  const [query, setQuery] = useState<Dict>({ basis: 'latest_full' });
  const [draft, setDraft] = useState<Dict>({ basis: 'latest_full' });
  const [sort, setSort] = useState({ key: 'name', descending: false });
  const [page, setPage] = useState(1);
  const [columns, setColumns] = useState(initialColumns);
  const [showColumns, setShowColumns] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => { setQuery(prev => ({ ...prev, search })); setPage(1); }, 250);
    return () => clearTimeout(timer);
  }, [search]);
  const response = useApi<Catalog>(url('/api/catalog', { ...query, sort_by: sort.key, descending: sort.descending,
    page, page_size: 50, revision }));
  const metrics = response.data?.metrics || [];
  const visible = metrics.filter(metric => columns.includes(metric.key));
  const categories = [...new Set(factors.map(f => f.category).filter(Boolean))];
  function apply() {
    const conditions: Dict = {};
    for (const [key, metric, op, divisor] of [['min_ic_ir', 'ic_ir', 'gte', 1], ['min_sharpe', 'sharpe', 'gte', 1],
      ['min_annual', 'annual_return', 'gte', 100], ['max_drawdown', 'max_drawdown', 'lte', 100]] as const) {
      if (draft[key] !== '' && draft[key] !== undefined) conditions[metric] = { [op]: Number(draft[key]) / divisor };
    }
    const { min_ic_ir, min_sharpe, min_annual, max_drawdown, ...filters } = draft;
    setQuery({ ...filters, search, metrics: Object.keys(conditions).length ? JSON.stringify(conditions) : '' });
    setPage(1);
  }
  function sortColumn(key: string) { setSort(prev => ({ key, descending: prev.key === key ? !prev.descending : key !== 'name' && key !== 'category' })); setPage(1); }
  function sortable(key: string, label: string) {
    return <button className="column-sort" onClick={() => sortColumn(key)}>{label}{sort.key === key ? sort.descending ? <ArrowDown size={13}/> : <ArrowUp size={13}/> : <ArrowUpDown size={12}/>}</button>;
  }
  function exportCsv() {
    if (!response.data) return;
    const head = ['因子', '标识', '分类', '实验数', ...visible.map(m => m.label), '区间', '指标来源', '来源实验'];
    const rows = response.data.items.map(r => [r.name, r.key, r.category, r.experiment_count,
      ...visible.map(m => r.metrics[m.key]), range(r.source?.period), r.metric_basis, r.source?.experiment_id || '']);
    const quote = (value: unknown) => '"' + String(value ?? '').replaceAll('"', '""') + '"';
    const blob = new Blob(['\ufeff' + [head, ...rows].map(row => row.map(quote).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' });
    const href = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = href; link.download = '因子指标.csv'; link.click(); URL.revokeObjectURL(href);
  }
  return <>
    <form className="panel catalog-filters" onSubmit={e => { e.preventDefault(); apply(); }}>
      <div className="catalog-search"><Search size={16}/><input aria-label="搜索因子" placeholder="搜索因子代码 / 中文名" value={search} onChange={e => setSearch(e.target.value)}/></div>
      <label>分类<select value={draft.category || ''} onChange={e => setDraft({ ...draft, category: e.target.value })}><option value="">全部</option>{categories.map(c => <option key={c}>{c}</option>)}</select></label>
      <label>条目<select value={draft.entry_type || ''} onChange={e => setDraft({ ...draft, entry_type: e.target.value })}><option value="">全部条目</option><option value="root">研究母因子</option><option value="derived">派生条目</option></select></label>
      <label>策略<select value={draft.strategy_type || ''} onChange={e => setDraft({ ...draft, strategy_type: e.target.value })}><option value="">全部</option><option value="cross_sectional">截面</option><option value="timing">择时</option></select></label>
      <label>指标来源<select aria-label="指标来源" value={draft.basis} onChange={e => setDraft({ ...draft, basis: e.target.value })}><option value="latest_full">最新完整实验</option><option value="latest">最新入库记录</option></select></label>
      <label>调仓频率<select value={draft.frequency || ''} onChange={e => setDraft({ ...draft, frequency: e.target.value })}><option value="">全部</option>{(response.data?.frequencies || []).map(f => <option key={f}>{f}</option>)}</select></label>
      <label>股票池<select value={draft.universe || ''} onChange={e => setDraft({ ...draft, universe: e.target.value })}><option value="">全部</option>{(response.data?.universes || []).map(f => <option key={f}>{f}</option>)}</select></label>
      <label>ICIR ≥<input aria-label="最小ICIR" type="number" step="any" value={draft.min_ic_ir ?? ''} onChange={e => setDraft({ ...draft, min_ic_ir: e.target.value })} placeholder="不限"/></label>
      <label>夏普 ≥<input aria-label="最小夏普" type="number" step="any" value={draft.min_sharpe ?? ''} onChange={e => setDraft({ ...draft, min_sharpe: e.target.value })} placeholder="不限"/></label>
      <label>年化收益 ≥ %<input type="number" step="any" value={draft.min_annual ?? ''} onChange={e => setDraft({ ...draft, min_annual: e.target.value })} placeholder="不限"/></label>
      <label>最大回撤 ≤ %<input type="number" min="0" step="any" value={draft.max_drawdown ?? ''} onChange={e => setDraft({ ...draft, max_drawdown: e.target.value })} placeholder="不限"/></label>
      <div className="catalog-actions"><button type="button" onClick={() => { setDraft({ basis: 'latest_full' }); setQuery({ basis: 'latest_full' }); setSearch(''); setPage(1); }}>重置</button><button className="primary" type="submit">应用筛选</button></div>
    </form>
    <section className="panel catalog-panel"><div className="row spread section-heading"><div><h3>因子评价指标 <small>{response.data?.total || 0} 个因子条目</small></h3><p className="muted">每行明确对应一条已归档实验；无完整快照时使用最新指标记录，缺失值显示为 —。</p></div><div className="row"><button onClick={() => setShowColumns(v => !v)}><Columns3 size={14}/>显示列</button><button onClick={exportCsv} disabled={!response.data?.items.length}><ArrowDownToLine size={14}/>导出本页 CSV</button></div></div>
      {showColumns && <div className="catalog-columns">{metrics.map(m => <label key={m.key}><input type="checkbox" checked={columns.includes(m.key)} onChange={() => setColumns(prev => prev.includes(m.key) ? prev.filter(k => k !== m.key) : [...prev, m.key])}/>{m.label}</label>)}</div>}
      {response.error && <div role="alert" className="error">{response.error}</div>}
      <div className="table-wrap"><table className="factor-metrics-table"><thead><tr><th aria-sort={sort.key === 'name' ? sort.descending ? 'descending' : 'ascending' : 'none'}>{sortable('name', '因子')}</th><th>{sortable('category', '类别')}</th><th>{sortable('experiment_count', '实验数')}</th>{visible.map(m => <th key={m.key} aria-sort={sort.key === m.key ? sort.descending ? 'descending' : 'ascending' : 'none'}>{sortable(m.key, m.label)}</th>)}<th>回测区间</th><th>指标口径 / 来源</th><th/></tr></thead>
        <tbody>{(response.data?.items || []).map(row => <tr key={row.id}><td><button className="factor-name-link" onClick={() => onChoose(row, row.source?.experiment_id)}>{row.name}</button><small className="block mono">{row.key} {row.parent_id && <span className="badge green">派生</span>}</small></td><td>{row.category || '—'}</td><td>{row.experiment_count}</td>{visible.map(m => <td className="catalog-number" key={m.key} title={row.metric_keys[m.key] || '未归档此项指标'}>{valueText(row.metrics[m.key], m.format)}</td>)}<td className="nowrap">{range(row.source?.period)}</td><td><span className="badge">{row.metric_basis}</span><small className="block mono" title={row.source?.experiment_id}>{row.source ? `${row.source.experiment_id.slice(0, 8)} · ${timestamp(row.source.created_at)}` : '无实验'}</small></td><td><button aria-label={`查看因子 ${row.name}`} className="icon-button" onClick={() => onChoose(row, row.source?.experiment_id)}><ChevronRight size={16}/></button></td></tr>)}</tbody>
      </table></div>
      {!response.loading && !response.data?.items.length && <div className="empty"><h3>暂无匹配的因子条目</h3><p>可调整筛选条件，或归档新的实验结果。</p></div>}
      {response.loading && <p className="loading">读取因子指标…</p>}
      <div className="pager"><span>共 {response.data?.total || 0} 个条目</span><div><button aria-label="因子上一页" disabled={page <= 1} onClick={() => setPage(p => p - 1)}><ArrowLeft size={14}/></button><span>{page} / {Math.max(1, Math.ceil((response.data?.total || 0) / 50))}</span><button aria-label="因子下一页" disabled={page * 50 >= (response.data?.total || 0)} onClick={() => setPage(p => p + 1)}><ArrowRight size={14}/></button></div></div>
    </section>
  </>;
}

function ResultSurface({ identity, scope, isTiming }: { identity: string; scope: string; isTiming: boolean }) {
  const [target, setTarget] = useState<ShadowRoot | null>(null);
  const [tab, setTab] = useState<ResultTab>('overview');
  const response = useApi<RunResult & { availability: Dict }>(url(`/api/experiments/${identity}/result`, { scope }));
  const client = useMemo<BacktestResultClient>(() => ({
    profiling: (id, options = {}) => getJson(url(`/api/experiments/${id}/profiling`, { scope, date_from: options.dateFrom, date_to: options.dateTo })),
    table: (id, kind, options = {}) => getJson(url(`/api/experiments/${id}/result-table/${kind}`, { scope,
      page: options.page || 1, size: options.size || 50, query: options.query, date_from: options.dateFrom, date_to: options.dateTo,
      ...Object.fromEntries(Object.entries(options.filters || {}).map(([key, value]) => ['filter.' + key, value])) })),
    downloadUrl: (id, kind) => url(`/api/experiments/${id}/result-download/${kind}`, { scope }),
  }), [scope]);
  // Import the actual Dashboard stylesheet. Shadow DOM confines its global
  // rules to this result surface while preserving future style updates.
  const styles = dashboardStyles.replaceAll(':root', ':host') + '\n:host{display:block;min-width:0;width:100%} .detail-layout{min-height:600px}.side-nav{padding-top:20px}.content-panel{min-width:0}';
  return <>
    {response.error && <div role="alert" className="error">{response.error}</div>}
    {response.loading && <p className="loading">加载归档回测结果…</p>}
    {response.data?.availability.raw === false && <div className="archive-notice">该实验仅保存评价指标，未归档净值及持仓；图表和原始明细按实际可用数据展示。</div>}
    <div className="dashboard-result-surface" ref={node => { if (node && !node.shadowRoot) setTarget(node.attachShadow({ mode: 'open' })); }}>
      {target && response.data && createPortal(<><style>{styles}</style><div className="detail-layout"><ResultNavigation activeTab={tab} onTab={setTab} isTiming={isTiming}/><section className="content-panel"><BacktestResultView result={response.data} state={response.data.run} runId={identity} tab={tab} isTiming={isTiming} client={client}/></section></div></>, target)}
    </div>
  </>;
}

export function FactorHistory({ factor, initialId, onBack, onInspect }: {
  factor: Dict; initialId?: string; onBack: () => void; onInspect: (record: any) => void;
}) {
  const [page, setPage] = useState(1);
  const [records, setRecords] = useState<Dict[]>([]);
  const [selected, setSelected] = useState(initialId || '');
  const [scope, setScope] = useState('total');
  const history = useApi<{ items: Dict[]; total: number }>(url(`/api/factors/${factor.id}/history`, { page, page_size: 200 }));
  useEffect(() => { if (history.data) { setRecords(prev => [...prev, ...history.data!.items.filter(r => !prev.some(p => p.id === r.id))]); if (!selected && history.data.items.length) setSelected(history.data.items[0].id); } }, [history.data]);
  const experiment = useApi<Dict>(selected ? `/api/experiments/${selected}` : null);
  const scopes: string[] = experiment.data?.metadata.available_scopes?.length ? experiment.data.metadata.available_scopes : ['total'];
  const activeScope = scopes.includes(scope) ? scope : scopes[0];
  const choices = [...records];
  if (experiment.data && !choices.some(r => r.id === selected)) choices.unshift({ ...experiment.data, period: {} });
  return <>
    <div className="factor-detail-heading"><button onClick={onBack}><ArrowLeft size={14}/>返回因子列表</button><div><span className="badge green">{factor.parent_id ? '派生条目' : '研究母因子'}</span><code>{factor.key}</code></div></div>
    <div className="panel history-selector"><label>历次实验<select aria-label="历次实验" value={selected} onChange={e => { setSelected(e.target.value); setScope('total'); }}>{choices.map(record => <option key={record.id} value={record.id}>{timestamp(record.created_at)} · {record.kind} · {record.has_raw ? '含原始结果' : '仅指标'} · {record.id.slice(0, 8)} · {paramsText(record.parameters).slice(0, 100)}</option>)}</select></label><span className="muted">共 {history.data?.total || 0} 次实验</span>{records.length < (history.data?.total || 0) && <button disabled={history.loading} onClick={() => setPage(p => p + 1)}>加载更早记录</button>}{scopes.length > 1 && <label>结果范围<select aria-label="结果范围" value={activeScope} onChange={e => setScope(e.target.value)}>{scopes.map(s => <option key={s}>{s}</option>)}</select></label>}{experiment.data && <button onClick={() => onInspect(experiment.data)}>原始数据与配置</button>}</div>
    {factor.description && <p className="muted">{factor.description}</p>}
    {Object.keys(factor.locks || {}).length > 0 && <div className="factor-locks">{Object.entries(factor.locks).map(([key, value]) => <span className="param" key={key}>{key} = {JSON.stringify(value)}</span>)}</div>}
    {(history.error || experiment.error) && <div className="error" role="alert">{history.error || experiment.error}</div>}
    {!selected && !history.loading && <div className="empty"><h3>该因子暂无已归档实验</h3></div>}
    {experiment.data?.capture_error && <div className="error">原始结果捕获异常：{experiment.data.capture_error}</div>}
    {experiment.data && <div className="history-source-note"><span>实验 {selected.slice(0, 8)}</span><span>{paramsText(experiment.data.parameters)}</span><span>{experiment.data.metadata.metric_basis || '归档回测指标'}</span></div>}
    {experiment.data && <ResultSurface key={selected + ':' + activeScope} identity={selected} scope={activeScope} isTiming={factor.strategy_type === 'timing'}/>}
  </>;
}
