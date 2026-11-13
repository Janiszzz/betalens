import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import { Activity, ArrowDownToLine, ArrowLeft, ArrowRight, BarChart3, Check, ChevronRight,
  Database, FlaskConical, GitCompareArrows, Layers3, Library, RefreshCw, Search, SlidersHorizontal, X } from 'lucide-react';
import './styles.css';
import { url, useApi } from './http';
import { FactorCatalog, FactorHistory } from './FactorBrowser';

const Plot = createPlotlyComponent(Plotly);
type Dict = Record<string, any>;
type Factor = { id: string; key: string; name: string; category: string; description: string;
  parent_id: string | null; locks: Dict; strategy_type: string; metadata: Dict };
type Experiment = { id: string; factor_id: string; kind: string; created_at: string;
  batch_id: string | null; parameters: Dict; metrics: Dict; metadata: Dict; config: Dict;
  tables: Record<string, { rows: number; columns: string[]; kind: string }>; has_raw: boolean;
  capture_error: string | null; tags: string[]; factor_snapshot?: Dict };
type Page = { items: Experiment[]; total: number; page: number; page_size: number };
type View = 'factors' | 'factor' | 'experiments' | 'mining' | 'compare';

const short = (id: string) => id.slice(0, 8);
const time = (value: string) => new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Shanghai',
  dateStyle: 'short', timeStyle: 'short' }).format(new Date(value));
const cell = (value: any): string => value === null || value === undefined ? '—'
  : typeof value === 'number' ? Number.isInteger(value) ? value.toLocaleString('zh-CN') : value.toFixed(4)
  : typeof value === 'object' ? JSON.stringify(value) : String(value);
function metric(metrics: Dict, names: string[]) { return names.map(key => metrics[key]).find(value => value !== undefined); }
function percent(value: any) { return typeof value === 'number' ? (value * 100).toFixed(2) + '%' : '—'; }
function ErrorBox({ error }: { error: string }) { return error ? <div role="alert" className="error">{error}</div> : null; }
function Empty({ text = '这里还没有实验记录', note = '通过研究脚本归档后，结果会出现在这里。' }) {
  return <div className="empty"><FlaskConical size={34}/><h3>{text}</h3><p>{note}</p></div>;
}
function Json({ value }: { value: any }) { return <pre className="json">{JSON.stringify(value, null, 2)}</pre>; }
const palette = ['#1b7a68', '#d89b52', '#5687ac', '#b36876', '#8572ad', '#637d5b'];
function Chart({ data, title, height = 310, extra = {} }: { data: any[]; title?: string; height?: number; extra?: Dict }) {
  return <div className="chart">{title && <h3>{title}</h3>}<Plot data={data} layout={{
    autosize: true, height, margin: { t: 12, r: 22, b: 44, l: 60 },
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', colorway: palette,
    font: { family: 'Inter, system-ui, sans-serif', color: '#66736e', size: 11 },
    xaxis: { gridcolor: '#f0f2ed', zeroline: false }, yaxis: { gridcolor: '#edf0eb', zerolinecolor: '#d9dfd9' },
    legend: { orientation: 'h', y: 1.15, x: 0 }, hovermode: 'x unified', ...extra,
  } as any} config={{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }}
    style={{ width: '100%' }} useResizeHandler/></div>;
}
function SimpleTable({ rows }: { rows: Dict[] }) {
  const columns = Object.keys(rows[0] || {});
  return <div className="table-wrap"><table><thead><tr>{columns.map(c => <th key={c}>{c}</th>)}</tr></thead>
    <tbody>{rows.map((row, i) => <tr key={i}>{columns.map(c => <td key={c} title={cell(row[c])}>{cell(row[c])}</td>)}</tr>)}</tbody></table></div>;
}
function Extra({ value, name = '' }: { value: any; name?: string }) {
  if (value?.type === 'image') return <section className="panel"><h3>{name}</h3><img className="artifact-image" src={'data:image/png;base64,' + value.data} alt={name}/></section>;
  if (value?.type === 'table') return <section className="panel"><h3>{name} <small>预览 {value.rows.length} / {value.total} 行</small></h3><SimpleTable rows={value.rows}/></section>;
  if (value?.data && Array.isArray(value.data) && value.layout) return <Chart data={value.data} title={name} extra={value.layout}/>;
  if (value && typeof value === 'object' && !Array.isArray(value)) return <>{Object.entries(value).map(([key, item]) => <Extra key={key} value={item} name={name ? name + ' / ' + key : key}/>)}</>;
  return <section className="panel"><h3>{name}</h3><Json value={value}/></section>;
}
function Pager({ page, total, size, onChange }: { page: number; total: number; size: number; onChange: (v: number) => void }) {
  const pages = Math.max(1, Math.ceil(total / size));
  return <div className="pager"><span>共 {total.toLocaleString()} 条记录</span><div>
    <button disabled={page <= 1} onClick={() => onChange(page - 1)} aria-label="上一页"><ArrowLeft size={15}/></button>
    <span>{page} / {pages}</span><button disabled={page >= pages} onClick={() => onChange(page + 1)} aria-label="下一页"><ArrowRight size={15}/></button>
  </div></div>;
}

function RawTables({ experiment }: { experiment: Experiment }) {
  const names = Object.keys(experiment.tables);
  const [name, setName] = useState(names.find(n => n === 'rebalance_log') || names[0] || '');
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [draft, setDraft] = useState('');
  const [columns, setColumns] = useState('');
  const [filters, setFilters] = useState('');
  useEffect(() => {
    setPage(1); setSearch(''); setDraft(''); setFilters('');
    const descriptor = experiment.tables[name];
    setColumns(descriptor?.kind === 'frame' ? descriptor.columns.slice(0, 12).join(',') : '');
  }, [name, experiment.id]);
  const response = useApi<Dict>(name ? url(`/api/experiments/${experiment.id}/table`, { name, page, page_size: 50, search, columns, filters }) : null);
  return <section className="panel"><div className="row spread"><div><h3>原始数据表</h3><p className="muted">保留原始时间索引与数值，下载包含完整列。</p></div><div className="row">
    {['csv', 'parquet'].map(format => <a key={format} className="button" href={url(`/api/experiments/${experiment.id}/download`, { name, format })}><ArrowDownToLine size={14}/>{format.toUpperCase()}</a>)}
  </div></div><label>数据表<select value={name} onChange={e => setName(e.target.value)}>{names.map(n => <option key={n} value={n}>{n} · {experiment.tables[n].rows.toLocaleString()} 行</option>)}</select></label>
    <form className="raw-filters" onSubmit={e => { e.preventDefault(); setPage(1); setSearch(draft); }}>
      <label>搜索内容<input value={draft} onChange={e => setDraft(e.target.value)} placeholder="证券代码、日期或文本"/></label>
      {experiment.tables[name]?.kind === 'frame' && <label>显示列（逗号分隔）<input aria-label="显示列" value={columns} onChange={e => { setColumns(e.target.value); setPage(1); }}/></label>}
      <label>列条件（JSON）<input value={filters} onChange={e => { setFilters(e.target.value); setPage(1); }} placeholder={'{"code":"000001.SZ"}'}/></label>
      <button className="primary" type="submit">筛选</button>
    </form><ErrorBox error={response.error}/>{response.loading && <p className="muted">读取数据…</p>}
    {response.data && <><SimpleTable rows={response.data.rows}/><Pager page={page} total={response.data.total} size={50} onChange={setPage}/></>}
  </section>;
}

function ExperimentDetail({ experiment, factor, onClose }: { experiment: Experiment; factor?: Factor; onClose: () => void }) {
  const [tab, setTab] = useState('tables');
  const extras = useApi<Dict>(tab === 'research' ? `/api/experiments/${experiment.id}/extras` : null);
  useEffect(() => {
    const close = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', close);
    const old = document.body.style.overflow; document.body.style.overflow = 'hidden';
    return () => { document.removeEventListener('keydown', close); document.body.style.overflow = old; };
  }, [onClose]);
  return <div className="modal-backdrop" onClick={onClose}><section className="detail" role="dialog" aria-modal="true" aria-label="原始实验数据" onClick={e => e.stopPropagation()}>
    <header className="detail-header"><div><div className="eyebrow">EXPERIMENT / {short(experiment.id)}</div><h2>{factor?.name || '原始实验数据'}</h2><p>{time(experiment.created_at)} · {experiment.kind}</p></div><button className="icon-button" onClick={onClose} aria-label="关闭详情"><X/></button></header>
    <nav className="tabs">{[['tables', '原始数据'], ['config', '参数与来源'], ['research', '研究产物']].map(([key, title]) => <button key={key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>{title}</button>)}</nav>
    <div className="detail-body">
      {tab === 'tables' && <RawTables experiment={experiment}/>}
      {tab === 'config' && <><section className="panel"><h3>实验参数</h3><Json value={experiment.parameters}/></section><section className="panel"><h3>完整有效配置</h3><Json value={experiment.config}/></section><section className="panel"><h3>归档因子元信息</h3><Json value={experiment.factor_snapshot}/></section><section className="panel"><h3>来源与计算版本</h3><Json value={experiment.metadata}/></section></>}
      {tab === 'research' && <><ErrorBox error={extras.error}/>{extras.loading && <p>读取研究产物…</p>}{extras.data && Object.keys(extras.data).length ? <Extra value={extras.data}/> : !extras.loading && <Empty text="没有附加研究产物" note="已有因子值、IC 和研究数据可在原始数据页查看。"/>}</>}
    </div></section></div>;
}

function ExperimentTable({ records, factorMap, selected, toggle, open }: { records: Experiment[]; factorMap: Record<string, Factor>; selected: string[]; toggle: (id: string) => void; open: (e: Experiment) => void }) {
  return <div className="table-wrap"><table className="experiments-table"><thead><tr><th className="check-cell">对比</th><th>因子 / 实验</th><th>参数</th><th>年化收益</th><th>夏普</th><th>目标值</th><th>最大回撤</th><th>数据</th><th>入库时间</th><th/></tr></thead>
    <tbody>{records.map(e => <tr key={e.id}><td><input aria-label={'选择实验 ' + short(e.id)} type="checkbox" checked={selected.includes(e.id)} onChange={() => toggle(e.id)}/></td>
      <td><button className="text-link" onClick={() => open(e)}>{factorMap[e.factor_id]?.name || '未命名因子'}</button><small className="block mono">{short(e.id)} · {e.kind}{e.metadata.selected ? ' · 入选' : ''}</small></td>
      <td className="parameter-cell" title={JSON.stringify(e.parameters)}>{Object.entries(e.parameters).slice(0, 4).map(([k, v]) => <span className="param" key={k}>{k} = {cell(v)}</span>)}{Object.keys(e.parameters).length === 0 && <span className="muted">见有效配置</span>}</td>
      <td>{percent(metric(e.metrics, ['年化收益', 'ann_ret_median', 'ann_ret']))}</td><td className="metric-value">{cell(metric(e.metrics, ['夏普比率', 'sharpe_median', 'sharpe']))}</td><td title={JSON.stringify(e.metadata.objective || {})}>{cell(e.metrics.objective)}</td><td>{percent(metric(e.metrics, ['最大回撤', 'max_dd_median', 'max_dd']))}</td>
      <td><span className={'badge ' + (e.capture_error ? 'amber' : e.has_raw ? 'green' : '')}>{e.capture_error ? '捕获异常' : e.has_raw ? '含原始表' : '仅指标'}</span></td><td className="muted nowrap">{time(e.created_at)}</td><td><button className="icon-button" aria-label={'查看实验 ' + short(e.id)} onClick={() => open(e)}><ChevronRight size={17}/></button></td>
    </tr>)}</tbody></table></div>;
}

function Compare({ ids, open }: { ids: string[]; open: (e: Experiment) => void }) {
  const [scope, setScope] = useState('total');
  const data = useApi<Dict[]>(ids.length ? url('/api/compare', { ids: ids.join(','), scope }) : null);
  const scopes: string[] = [...new Set(['total', ...(data.data || []).flatMap(r => r.experiment.metadata.available_scopes || [])])];
  const metricNames = [...new Set((data.data || []).flatMap(r => Object.keys(r.experiment.metrics)))];
  return <><ErrorBox error={data.error}/>{!ids.length && <Empty text="选择实验开始比较" note="在实验记录中勾选需要比较的实验，可跨因子、跨批次选择。"/>}
    {data.loading && <p>读取对比结果…</p>}{data.data && <><div className="row spread"><p className="muted">仅绘制所选范围中已捕获的净值；全部记录仍参与指标比较。</p><label className="scope-label">结果范围<select value={scope} onChange={e => setScope(e.target.value)}>{scopes.map(s => <option key={s} value={s}>{s === 'total' ? '全程' : s}</option>)}</select></label></div><Chart title="净值路径对比" height={380} data={data.data.filter(r => r.charts).map(r => ({ x: r.charts.nav.map((p: Dict) => p.time), y: r.charts.nav.map((p: Dict) => p.value), name: r.factor_name + ' · ' + short(r.experiment.id), type: 'scatter', mode: 'lines', connectgaps: false }))}/>
      <section className="panel"><h3>实验口径</h3><p className="muted">净值按保存时的时间轴展示。窗口聚合指标与整段回测指标分别保留，请结合评价区间比较。</p><div className="comparison-cards">{data.data.map(r => <button className="comparison-card" key={r.experiment.id} onClick={() => open(r.experiment)}><b>{r.factor_name}</b><small>{short(r.experiment.id)} · {r.experiment.has_raw ? '包含原始结果' : '仅评价指标'}</small><span>{cell(r.experiment.parameters)}</span></button>)}</div></section>
      <section className="panel"><h3>归档指标对比</h3><SimpleTable rows={metricNames.map(指标 => ({ 指标, ...Object.fromEntries(data.data!.map(r => [r.factor_name + ' / ' + short(r.experiment.id), r.experiment.metrics[指标]])) }))}/></section></>}
  </>;
}

function App() {
  const [view, setView] = useState<View>('factors');
  const [factorId, setFactorId] = useState('');
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<string[]>([]);
  const [detail, setDetail] = useState<Experiment>();
  const [historyInitial, setHistoryInitial] = useState<string>();
  const [catalogRevision, setCatalogRevision] = useState(0);
  const [query, setQuery] = useState<Dict>({});
  const [draft, setDraft] = useState<Dict>({});
  const [batchId, setBatchId] = useState('');
  const [heat, setHeat] = useState<Dict>({ metric: 'sharpe' });
  useEffect(() => { setHeat(previous => ({ metric: previous.metric || 'sharpe' })); }, [factorId, batchId]);
  const factors = useApi<Factor[]>('/api/factors');
  const batches = useApi<Dict[]>('/api/batches');
  const health = useApi<Dict>('/api/health');
  const factorMap = useMemo(() => Object.fromEntries((factors.data || []).map(f => [f.id, f])), [factors.data]);
  const selectedFactor = factorMap[factorId];
  const records = useApi<Page>(view === 'experiments' || view === 'mining' ? url('/api/experiments', {
    ...query, factor_id: factorId, page, page_size: 30, kind: view === 'mining' ? 'mining' : query.kind,
    batch_id: view === 'mining' ? batchId : query.batch_id, sort_by: view === 'mining' ? 'objective' : '',
  }) : null);
  const heatmap = useApi<Dict>(view === 'mining' ? url('/api/mining/heatmap', { factor_id: factorId, batch_id: batchId, ...heat }) : null);
  const activeBatch = (batches.data || []).find(b => b.id === batchId);
  function navigate(next: View) { setView(next); setPage(1); setQuery({}); setDraft({}); }
  function toggle(id: string) { setSelected(v => v.includes(id) ? v.filter(s => s !== id) : v.length < 20 ? [...v, id] : v); }
  function chooseFactor(f: Factor, experimentId?: string) { setFactorId(f.id); setHistoryInitial(experimentId); navigate('factor'); }
  function openExperiment(e: Experiment) { setFactorId(e.factor_id); setHistoryInitial(e.id); navigate('factor'); }
  const titles: Record<View, string> = { factors: '因子库', factor: selectedFactor?.name || '因子详情', experiments: '实验记录', mining: '参数挖掘', compare: '实验对比' };
  const descriptions: Record<View, string> = { factors: '筛选与排序评价指标，进入因子查看历次实验的完整回测结果。', factor: '切换历次实验，查看与 Dashboard 同步的回测结果。', experiments: '浏览已归档的研究结果，按条件筛选，选择实验进行比较。', mining: '回看搜索阶段与窗口表现，观察参数变化带来的结果。', compare: '将不同因子、参数与批次放在同一视野中。' };
  return <div className="app"><aside className="sidebar"><a className="brand" href="#" onClick={e => { e.preventDefault(); navigate('factors'); }}><span className="brand-mark">b<span>.</span></span><div><strong>betalens</strong><small>RESEARCH ARCHIVE</small></div></a>
    <div className="nav-label">研究工作区</div><nav className="main-nav">{([
      ['factors', Library, '因子库'], ['experiments', FlaskConical, '实验记录'], ['mining', Layers3, '参数挖掘'], ['compare', GitCompareArrows, '实验对比'],
    ] as const).map(([key, Icon, label]) => <button key={key} className={view === key || view === 'factor' && key === 'factors' ? 'active' : ''} onClick={() => navigate(key)}><Icon size={18}/>{label}{key === 'compare' && selected.length > 0 && <span className="nav-count">{selected.length}</span>}</button>)}</nav>
    <div className="sidebar-scope"><div className="nav-label">浏览范围</div><select aria-label="浏览因子范围" value={factorId} onChange={e => { setFactorId(e.target.value); setHistoryInitial(undefined); setPage(1); if (view === 'factor' && !e.target.value) setView('factors'); }}><option value="">全部因子</option>{(factors.data || []).map(f => <option key={f.id} value={f.id}>{f.name}{f.parent_id ? ' · 派生条目' : ''}</option>)}</select>{selectedFactor && <small>{selectedFactor.description || selectedFactor.key}</small>}</div>
    <div className="sidebar-footer"><span className="status-dot"/> 本地只读档案<small>实验由研究脚本发起<br/>此处只展示已入库结果</small></div>
  </aside><div className="workspace"><header className="topbar"><div><span className="muted">研究档案</span><ChevronRight size={13}/><span>{titles[view]}</span></div><div className="row"><span className={'connection ' + (health.error ? 'offline' : '')}><span className="status-dot"/>{health.error ? '连接失败' : '本地实验库'}</span><button className="icon-button" aria-label="刷新" onClick={() => { factors.refresh(); setCatalogRevision(v => v + 1); batches.refresh(); records.refresh(); heatmap.refresh(); health.refresh(); }}><RefreshCw size={16}/></button></div></header>
    <main><div className="page-heading"><div><div className="eyebrow">BETALENS / RESEARCH</div><h1>{titles[view]}</h1><p>{descriptions[view]}</p></div><span className="page-icon">{view === 'factors' ? <Library/> : view === 'mining' ? <BarChart3/> : view === 'compare' ? <GitCompareArrows/> : <FlaskConical/>}</span></div>
      <ErrorBox error={factors.error || health.error}/>
      {view === 'factors' && <FactorCatalog factors={factors.data || []} revision={catalogRevision} onChoose={chooseFactor}/>}
      {view === 'factor' && selectedFactor && <FactorHistory key={factorId + ':' + (historyInitial || '')} factor={selectedFactor} initialId={historyInitial} onBack={() => navigate('factors')} onInspect={setDetail}/>}
      {(view === 'experiments' || view === 'mining') && <>
        {selectedFactor && <div className="context-bar"><span className="badge green">{selectedFactor.parent_id ? '派生条目' : '研究母因子'}</span><strong>{selectedFactor.name}</strong>{Object.entries(selectedFactor.locks).map(([k, v]) => <span className="param" key={k}>{k} = {cell(v)}</span>)}<button className="text-link" onClick={() => { setFactorId(''); setPage(1); }}>查看全部因子</button></div>}
        {view === 'experiments' && <form className="panel filter-form" onSubmit={e => { e.preventDefault(); setPage(1); setQuery({ ...draft, start: draft.start ? new Date(draft.start + 'T00:00:00+08:00').toISOString() : '', end: draft.end ? new Date(draft.end + 'T23:59:59.999+08:00').toISOString() : '' }); }}>
          <label>检索<input placeholder="参数、标签或实验标识" value={draft.search || ''} onChange={e => setDraft({ ...draft, search: e.target.value })}/></label><label>实验类型<select value={draft.kind || ''} onChange={e => setDraft({ ...draft, kind: e.target.value })}><option value="">全部类型</option><option value="manual">手工单次</option><option value="batch">多因子跑批</option><option value="mining">参数挖掘</option></select></label>
          <label>原始结果<select value={draft.has_raw ?? ''} onChange={e => setDraft({ ...draft, has_raw: e.target.value })}><option value="">全部记录</option><option value="true">包含原始结果</option><option value="false">仅评价指标</option></select></label><label>所属批次<select value={draft.batch_id || ''} onChange={e => setDraft({ ...draft, batch_id: e.target.value })}><option value="">全部批次</option>{(batches.data || []).map(b => <option key={b.id} value={b.id}>{short(b.id)} · {b.kind}</option>)}</select></label>
          <label>入库起始日期<input type="date" value={draft.start || ''} onChange={e => setDraft({ ...draft, start: e.target.value })}/></label><label>入库结束日期<input type="date" value={draft.end || ''} onChange={e => setDraft({ ...draft, end: e.target.value })}/></label><label>标签（逗号分隔）<input value={draft.tags || ''} onChange={e => setDraft({ ...draft, tags: e.target.value })}/></label>
          <details className="advanced"><summary>参数与指标条件</summary><div className="row"><label>参数条件（JSON）<input value={draft.parameters || ''} onChange={e => setDraft({ ...draft, parameters: e.target.value })} placeholder={'{"window":20}'}/></label><label>配置条件（字段路径）<input value={draft.config || ''} onChange={e => setDraft({ ...draft, config: e.target.value })} placeholder={'{"run.rebal_freq":"W"}'}/></label><label>指标条件（JSON）<input value={draft.metrics || ''} onChange={e => setDraft({ ...draft, metrics: e.target.value })} placeholder={'{"夏普比率":{"gte":1}}'}/></label></div></details><div className="filter-actions"><button type="button" onClick={() => { setDraft({}); setQuery({}); setPage(1); }}>重置</button><button type="submit" className="primary"><Search size={14}/>应用筛选</button></div>
        </form>}
        {view === 'mining' && <><section className="panel"><div className="row spread"><div><h3>参数表现热力图</h3><p className="muted">按搜索阶段与滑动窗口方案回看原始评价结果。</p></div><span className="badge green">窗口指标 · 均值聚合</span></div><div className="heat-filters">
          <label>挖掘批次<select value={batchId} onChange={e => { setBatchId(e.target.value); setPage(1); }}><option value="">全部挖掘批次</option>{(batches.data || []).filter(b => b.kind === 'mining').map(b => <option key={b.id} value={b.id}>{short(b.id)} · {time(b.created_at)}</option>)}</select></label>
          <label>横轴参数<select value={heat.x || heatmap.data?.x_parameter || ''} onChange={e => setHeat({ ...heat, x: e.target.value })}>{(heatmap.data?.parameters || []).map((p: string) => <option key={p}>{p}</option>)}</select></label><label>纵轴参数<select value={heat.y || heatmap.data?.y_parameter || '__all__'} onChange={e => setHeat({ ...heat, y: e.target.value })}><option value="__all__">全部</option>{(heatmap.data?.parameters || []).map((p: string) => <option key={p}>{p}</option>)}</select></label>
          <label>评价指标<select value={heat.metric} onChange={e => setHeat({ ...heat, metric: e.target.value })}>{[['sharpe', '夏普比率'], ['ann_ret', '年化收益'], ['calmar', '卡玛比率'], ['max_dd', '最大回撤'], ['rank_ic', 'Rank IC']].map(([k, n]) => <option key={k} value={k}>{n}</option>)}</select></label>
          <label>搜索阶段<select value={heat.stage || ''} onChange={e => setHeat({ ...heat, stage: e.target.value })}><option value="">全部阶段</option>{(heatmap.data?.stages || []).map((s: string) => <option key={s}>{s}</option>)}</select></label><label>窗口方案<select value={heat.window_scheme || ''} onChange={e => setHeat({ ...heat, window_scheme: e.target.value })}><option value="">全部窗口</option>{(heatmap.data?.window_schemes || []).map((s: string) => <option key={s}>{s}</option>)}</select></label></div>
          <ErrorBox error={heatmap.error}/>{heatmap.loading ? <p className="muted">聚合窗口结果…</p> : heatmap.data?.z.length ? <><Chart height={370} data={[{ type: 'heatmap', x: heatmap.data.x.map(String), y: heatmap.data.y.map(String), z: heatmap.data.z, customdata: heatmap.data.counts, colorscale: [[0, '#f2dfcf'], [0.5, '#f7f7ed'], [1, '#247b6a']], hovertemplate: '%{x} / %{y}<br>均值 %{z:.4f}<br>评价记录 %{customdata}<extra></extra>' }]} extra={{ xaxis: { title: { text: heatmap.data.x_parameter } }, yaxis: { title: { text: heatmap.data.y_parameter } } }}/><p className="muted fine-print">{heatmap.data.basis}</p></> : <Empty text="暂无可展示的参数热力图" note="选择已归档的挖掘批次或调整筛选范围。"/>}
          {activeBatch && <details className="batch-summary"><summary>批次状态：{activeBatch.status} · {activeBatch.experiment_ids.length} 条实验 · {activeBatch.failures.length} 项异常</summary><Json value={activeBatch.failures}/></details>}
        </section></>}
        <section className="panel records-panel"><div className="row spread section-heading"><h3>{view === 'mining' ? '候选排名' : '归档实验'} <small>{view === 'mining' ? '按上游目标方向排序，阶段与窗口口径见详情' : '按入库时间倒序'}</small></h3><span className="muted">已选 {selected.length} / 20</span></div><ErrorBox error={records.error}/>{records.loading && <p className="loading">读取实验记录…</p>}{records.data?.items.length ? <><ExperimentTable records={records.data.items} factorMap={factorMap} selected={selected} toggle={toggle} open={openExperiment}/><Pager page={page} total={records.data.total} size={30} onChange={setPage}/></> : !records.loading && <Empty/>}</section>
      </>}
      {view === 'compare' && <Compare ids={selected} open={openExperiment}/>}
    </main><footer className="workspace-footer"><span>Betalens · 因子实验档案</span><span>研究可追溯，结果可再分析</span></footer>
  </div>{selected.length > 0 && view !== 'compare' && <div className="compare-bar"><span><Check size={16}/>已选择 <strong>{selected.length}</strong> 条实验</span><button onClick={() => setSelected([])}>清空选择</button><button className="primary" onClick={() => navigate('compare')}><GitCompareArrows size={16}/>比较实验<ArrowRight size={15}/></button></div>}
    {detail && <ExperimentDetail key={detail.id} experiment={detail} factor={factorMap[detail.factor_id]} onClose={() => setDetail(undefined)}/>}
  </div>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
