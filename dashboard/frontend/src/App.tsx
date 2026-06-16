import { useEffect, useMemo, useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import {
  Activity,
  BarChart3,
  CheckCircle2,
  CircleDollarSign,
  ClipboardList,
  Download,
  FileText,
  Home,
  ListFilter,
  Loader2,
  Play,
  RotateCw,
  Search,
  Settings,
  Table2,
  TerminalSquare,
  XCircle
} from 'lucide-react';
import { api } from './api';
import type { FactorDetail, FactorSummary, Metric, RunResult, RunState } from './types';

type Page = 'home' | 'detail';
type ResultTab = 'overview' | 'trades' | 'positions' | 'logs';

const FREQ_LABELS: Record<string, string> = {
  D: '每天',
  W: '每周',
  ME: '月末',
  QE: '季末'
};

const STATUS_LABELS: Record<string, string> = {
  queued: '排队中',
  running: '运行中',
  completed: '回测完成',
  failed: '运行失败'
};

const formatValue = (metric: Metric) => {
  if (metric.value === null || metric.value === undefined || metric.value === '') return '-';
  if (typeof metric.value === 'string') return metric.value;
  if (metric.format === 'percent') return `${(metric.value * 100).toFixed(2)}%`;
  if (Number.isInteger(metric.value)) return String(metric.value);
  return Math.abs(metric.value) >= 100 ? metric.value.toFixed(2) : metric.value.toFixed(3);
};

const asString = (value: unknown, fallback = '') => {
  if (value === undefined || value === null) return fallback;
  return String(value);
};

const asBool = (value: unknown, fallback = false) => {
  if (typeof value === 'boolean') return value;
  return fallback;
};

const asNumber = (value: unknown, fallback: number) => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

function App() {
  const [page, setPage] = useState<Page>('home');
  const [factors, setFactors] = useState<FactorSummary[]>([]);
  const [selected, setSelected] = useState<FactorSummary | null>(null);
  const [detail, setDetail] = useState<FactorDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.factors()
      .then(setFactors)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const openFactor = async (factor: FactorSummary) => {
    setSelected(factor);
    setPage('detail');
    setError(null);
    try {
      const next = await api.factor(factor.factor_class, factor.name);
      setDetail(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="app-shell">
      <header className="top-strip">
        <button className="ghost-button" onClick={() => setPage('home')} title="主页">
          <Home size={17} />
        </button>
        <div>
          <div className="brand">betalens Dashboard</div>
          <div className="brand-sub">量化多因子回测控制台</div>
        </div>
      </header>
      {error && <div className="global-error">{error}</div>}
      {page === 'home' ? (
        <HomePage factors={factors} loading={loading} onOpen={openFactor} />
      ) : (
        <FactorPage factor={selected} detail={detail} onBack={() => setPage('home')} />
      )}
    </div>
  );
}

function HomePage({
  factors,
  loading,
  onOpen
}: {
  factors: FactorSummary[];
  loading: boolean;
  onOpen: (factor: FactorSummary) => void;
}) {
  const [query, setQuery] = useState('');
  const [factorClass, setFactorClass] = useState('全部');
  const classes = useMemo(() => ['全部', ...Array.from(new Set(factors.map((f) => f.factor_class)))], [factors]);
  const filtered = factors.filter((factor) => {
    const text = `${factor.factor_class} ${factor.name} ${factor.formula} ${factor.logic}`.toLowerCase();
    return (factorClass === '全部' || factor.factor_class === factorClass) && text.includes(query.toLowerCase());
  });

  return (
    <main className="home-page">
      <section className="home-toolbar">
        <div>
          <h1>因子回测</h1>
          <p>从 `betalens-factor` 自动发现因子，选择后配置参数并运行回测。</p>
        </div>
        <div className="home-filters">
          <div className="search-box">
            <Search size={16} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索因子/公式/逻辑" />
          </div>
          <select value={factorClass} onChange={(event) => setFactorClass(event.target.value)}>
            {classes.map((cls) => (
              <option key={cls}>{cls}</option>
            ))}
          </select>
        </div>
      </section>
      {loading ? (
        <div className="empty-state">
          <Loader2 className="spin" size={22} />
          正在扫描因子...
        </div>
      ) : (
        <section className="factor-grid">
          {filtered.map((factor) => (
            <button className="factor-card" key={`${factor.factor_class}/${factor.name}`} onClick={() => onOpen(factor)}>
              <div className="card-title-row">
                <span className="class-pill">{factor.factor_class}</span>
                <span className="freq-pill">{FREQ_LABELS[asString(factor.defaults.rebal_freq)] || asString(factor.defaults.rebal_freq, 'D')}</span>
              </div>
              <h2>{factor.name}</h2>
              <p className="formula">{factor.formula || '未提供公式'}</p>
              <p className="logic">{factor.logic || '未提供逻辑说明'}</p>
              <div className="input-list">
                {Object.entries(factor.inputs).map(([key, value]) => (
                  <span key={key}>{key}: {value}</span>
                ))}
              </div>
            </button>
          ))}
        </section>
      )}
    </main>
  );
}

function FactorPage({
  factor,
  detail,
  onBack
}: {
  factor: FactorSummary | null;
  detail: FactorDetail | null;
  onBack: () => void;
}) {
  const source = detail || factor;
  const defaults = source?.defaults || {};
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [computeKwargs, setComputeKwargs] = useState<Record<string, unknown>>({});
  const [runId, setRunId] = useState<string | null>(null);
  const [state, setState] = useState<RunState | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [logs, setLogs] = useState('');
  const [activeTab, setActiveTab] = useState<ResultTab>('overview');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!detail) return;
    setParams({
      start_date: asString(detail.defaults.start_date, '2024-01-01'),
      end_date: asString(detail.defaults.end_date, '2025-12-31'),
      initial_amount: asNumber(detail.defaults.initial_amount, 100000),
      rebal_freq: asString(detail.defaults.rebal_freq, 'D'),
      n_quantiles: asNumber(detail.defaults.n_quantiles, 20),
      index_code: asString(detail.defaults.index_code),
      direction: asString(detail.defaults.direction, 'positive'),
      use_industry: asBool(detail.defaults.use_industry),
      use_mktcap: asBool(detail.defaults.use_mktcap),
      industry_scheme: asString(detail.defaults.industry_scheme, '申万一级行业'),
      backtest_metric: asString(detail.defaults.backtest_metric, '收盘价(元)'),
      include_profiling: asBool(detail.defaults.include_profiling, true)
    });
    setComputeKwargs(detail.compute_kwargs || {});
  }, [detail]);

  useEffect(() => {
    if (!runId) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await api.run(runId);
        setState(next);
        if (next.status === 'completed') {
          const data = await api.result(runId);
          setResult(data);
          setActiveTab('overview');
          window.clearInterval(timer);
        }
        if (next.status === 'failed') {
          window.clearInterval(timer);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [runId]);

  useEffect(() => {
    if (!runId) return;
    setLogs('');
    const events = new EventSource(`/api/runs/${runId}/logs`);
    events.addEventListener('log', (event) => {
      const payload = JSON.parse((event as MessageEvent).data);
      setLogs((prev) => prev + payload.chunk);
    });
    events.addEventListener('close', () => events.close());
    events.onerror = () => events.close();
    return () => events.close();
  }, [runId]);

  const updateParam = (key: string, value: unknown) => setParams((prev) => ({ ...prev, [key]: value }));
  const updateCompute = (key: string, value: unknown) => setComputeKwargs((prev) => ({ ...prev, [key]: value }));

  const startRun = async () => {
    if (!source) return;
    setError(null);
    setResult(null);
    setLogs('');
    setState(null);
    try {
      const created = await api.startRun({
        factor_class: source.factor_class,
        name: source.name,
        parameters: params,
        compute_kwargs: computeKwargs
      });
      setRunId(created.run_id);
      setActiveTab('logs');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  if (!source) {
    return <main className="empty-state">未选择因子</main>;
  }

  return (
    <main className="detail-page">
      <section className="run-header">
        <div className="title-block">
          <button className="ghost-button" onClick={onBack}>返回</button>
          <div>
            <h1>{source.factor_class}/{source.name}</h1>
            <p>{source.source}</p>
          </div>
        </div>
        <div className="run-actions">
          <StatusBadge state={state} />
          <button className="primary-button" onClick={startRun} disabled={!detail || state?.status === 'running' || state?.status === 'queued'}>
            {state?.status === 'running' || state?.status === 'queued' ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
            运行回测
          </button>
        </div>
      </section>

      <section className="settings-bar">
        <span>设置：</span>
        <strong>{asString(params.start_date)} 到 {asString(params.end_date)}</strong>
        <span>￥{Number(params.initial_amount || 0).toLocaleString()}</span>
        <span>{FREQ_LABELS[asString(params.rebal_freq)] || asString(params.rebal_freq)}</span>
        <span>状态：{state ? STATUS_LABELS[state.status] : '未运行'}</span>
        {state?.elapsed_seconds ? <span>耗时 {state.elapsed_seconds.toFixed(1)}s</span> : null}
        <span className="python-pill">Python</span>
      </section>

      {error && <div className="global-error">{error}</div>}

      <div className="detail-layout">
        <aside className="side-nav">
          <NavButton icon={<CircleDollarSign size={18} />} active={activeTab === 'overview'} onClick={() => setActiveTab('overview')}>收益概述</NavButton>
          <NavButton icon={<ClipboardList size={18} />} active={activeTab === 'trades'} onClick={() => setActiveTab('trades')}>交易详情</NavButton>
          <NavButton icon={<BarChart3 size={18} />} active={activeTab === 'positions'} onClick={() => setActiveTab('positions')}>每日持仓&收益</NavButton>
          <NavButton icon={<TerminalSquare size={18} />} active={activeTab === 'logs'} onClick={() => setActiveTab('logs')}>日志输出</NavButton>
          <div className="nav-divider" />
          <ParameterPanel params={params} computeKwargs={computeKwargs} onParam={updateParam} onCompute={updateCompute} />
        </aside>

        <section className="content-panel">
          {activeTab === 'overview' && <Overview result={result} state={state} />}
          {activeTab === 'trades' && <Trades result={result} />}
          {activeTab === 'positions' && <Positions result={result} />}
          {activeTab === 'logs' && <Logs logs={logs} state={state} />}
        </section>
      </div>
    </main>
  );
}

function StatusBadge({ state }: { state: RunState | null }) {
  if (!state) return <span className="status-badge idle"><Activity size={16} />未运行</span>;
  const icon = state.status === 'completed' ? <CheckCircle2 size={16} /> : state.status === 'failed' ? <XCircle size={16} /> : <Loader2 className="spin" size={16} />;
  return <span className={`status-badge ${state.status}`}>{icon}{STATUS_LABELS[state.status]}</span>;
}

function NavButton({ icon, active, onClick, children }: { icon: React.ReactNode; active: boolean; onClick: () => void; children: React.ReactNode }) {
  return <button className={`nav-button ${active ? 'active' : ''}`} onClick={onClick}>{icon}<span>{children}</span></button>;
}

function ParameterPanel({
  params,
  computeKwargs,
  onParam,
  onCompute
}: {
  params: Record<string, unknown>;
  computeKwargs: Record<string, unknown>;
  onParam: (key: string, value: unknown) => void;
  onCompute: (key: string, value: unknown) => void;
}) {
  return (
    <div className="parameter-panel">
      <h3><Settings size={16} />参数</h3>
      <LabeledInput label="起始日期" type="date" value={asString(params.start_date)} onChange={(v) => onParam('start_date', v)} />
      <LabeledInput label="结束日期" type="date" value={asString(params.end_date)} onChange={(v) => onParam('end_date', v)} />
      <LabeledInput label="初始资金" type="number" value={asString(params.initial_amount)} onChange={(v) => onParam('initial_amount', Number(v))} />
      <label className="field">
        调仓频率
        <select value={asString(params.rebal_freq)} onChange={(event) => onParam('rebal_freq', event.target.value)}>
          <option value="D">每天</option>
          <option value="W">每周</option>
          <option value="ME">月末</option>
          <option value="QE">季末</option>
        </select>
      </label>
      <LabeledInput label="分组数" type="number" value={asString(params.n_quantiles)} onChange={(v) => onParam('n_quantiles', Number(v))} />
      <LabeledInput label="指数代码" value={asString(params.index_code)} onChange={(v) => onParam('index_code', v)} />
      <LabeledInput label="交易价格" value={asString(params.backtest_metric)} onChange={(v) => onParam('backtest_metric', v)} />
      <label className="field inline"><input type="checkbox" checked={Boolean(params.use_industry)} onChange={(event) => onParam('use_industry', event.target.checked)} />行业中性化</label>
      <label className="field inline"><input type="checkbox" checked={Boolean(params.use_mktcap)} onChange={(event) => onParam('use_mktcap', event.target.checked)} />市值中性化</label>
      <label className="field inline"><input type="checkbox" checked={Boolean(params.include_profiling)} onChange={(event) => onParam('include_profiling', event.target.checked)} />Profiling</label>
      {Object.keys(computeKwargs).length ? <h3><ListFilter size={16} />算子参数</h3> : null}
      {Object.entries(computeKwargs).map(([key, value]) => (
        <LabeledInput key={key} label={key} type={typeof value === 'number' ? 'number' : 'text'} value={asString(value)} onChange={(v) => onCompute(key, typeof value === 'number' ? Number(v) : v)} />
      ))}
    </div>
  );
}

function LabeledInput({ label, value, onChange, type = 'text' }: { label: string; value: string; onChange: (value: string) => void; type?: string }) {
  return <label className="field">{label}<input type={type} value={value} onChange={(event) => onChange(event.target.value)} /></label>;
}

function Overview({ result, state }: { result: RunResult | null; state: RunState | null }) {
  if (!result) return <Waiting state={state} />;
  const nav = result.charts.nav;
  const pnl = result.charts.dailyPnl;
  const positionSeries = buildPositionTraces(result.charts.positionValue);
  return (
    <div className="view-stack">
      <div className="section-title">收益概述</div>
      <div className="metrics-grid">
        {result.metrics.map((metric) => (
          <div className="metric-tile" key={metric.label}>
            <span>{metric.label}</span>
            <strong className={typeof metric.value === 'number' && metric.value < 0 ? 'negative' : ''}>{formatValue(metric)}</strong>
          </div>
        ))}
      </div>
      <div className="chart-card">
        <Plot
          data={[
            { x: nav.map((p) => p.date), y: nav.map((p) => p.nav), type: 'scatter', mode: 'lines', name: '策略净值', line: { color: '#2d66a8', width: 2 } }
          ]}
          layout={baseLayout('收益净值曲线', 360, true)}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          className="plot"
        />
      </div>
      <div className="chart-card">
        <Plot
          data={[
            { x: pnl.map((p) => p.date), y: pnl.map((p) => p.pnl), type: 'bar', name: '每日盈亏', marker: { color: pnl.map((p) => Number(p.pnl) >= 0 ? '#6a9f42' : '#8061a8') } }
          ]}
          layout={baseLayout('每日盈亏', 260, false)}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          className="plot"
        />
      </div>
      <div className="chart-card">
        <Plot
          data={positionSeries}
          layout={baseLayout('实时持仓叠加图', 300, false)}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          className="plot"
        />
      </div>
      <Downloads result={result} />
    </div>
  );
}

function baseLayout(title: string, height: number, slider: boolean) {
  return {
    title: { text: title, font: { size: 15 } },
    autosize: true,
    height,
    margin: { l: 46, r: 22, t: 42, b: slider ? 45 : 30 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    hovermode: 'x unified' as const,
    xaxis: { rangeslider: slider ? { visible: true, thickness: 0.08 } : undefined, gridcolor: '#d8dde3' },
    yaxis: { gridcolor: '#d8dde3', zerolinecolor: '#87909a' },
    showlegend: true,
    legend: { orientation: 'h' as const, x: 0, y: 1.12 }
  };
}

function buildPositionTraces(records: Array<Record<string, number | string>>) {
  const byCode = new Map<string, Array<Record<string, number | string>>>();
  records.forEach((record) => {
    const code = String(record.code);
    if (!byCode.has(code)) byCode.set(code, []);
    byCode.get(code)!.push(record);
  });
  return Array.from(byCode.entries()).slice(0, 12).map(([code, rows]) => ({
    x: rows.map((r) => r.date),
    y: rows.map((r) => r.value),
    type: 'scatter' as const,
    mode: 'lines' as const,
    stackgroup: 'one',
    name: code
  }));
}

function Downloads({ result }: { result: RunResult }) {
  const entries = Object.entries(result.downloads).filter(([, item]) => item.exists);
  if (!entries.length) return null;
  return (
    <div className="download-row">
      {entries.map(([kind]) => (
        <a key={kind} className="secondary-button" href={api.downloadUrl(result.run.run_id, kind)}>
          <Download size={15} />
          下载 {kind}
        </a>
      ))}
    </div>
  );
}

function Trades({ result }: { result: RunResult | null }) {
  const [query, setQuery] = useState('');
  const [direction, setDirection] = useState('全部');
  if (!result) return <Waiting state={null} />;
  const filtered = result.trades.filter((row) => {
    const text = JSON.stringify(row).toLowerCase();
    const dir = asString(row.direction);
    return text.includes(query.toLowerCase()) && (direction === '全部' || dir === direction);
  });
  return (
    <TableView
      title="交易详情"
      icon={<ClipboardList size={18} />}
      rows={filtered}
      controls={<><SearchInput value={query} onChange={setQuery} placeholder="搜索代码/字段" /><select value={direction} onChange={(e) => setDirection(e.target.value)}><option>全部</option><option value="buy">buy</option><option value="sell">sell</option></select></>}
    />
  );
}

function Positions({ result }: { result: RunResult | null }) {
  const [query, setQuery] = useState('');
  const [date, setDate] = useState('全部');
  if (!result) return <Waiting state={null} />;
  const dates = ['全部', ...Array.from(new Set(result.positions.map((row) => asString(row.date))))];
  const filtered = result.positions.filter((row) => {
    const text = JSON.stringify(row).toLowerCase();
    return text.includes(query.toLowerCase()) && (date === '全部' || asString(row.date) === date);
  });
  return (
    <TableView
      title="每日持仓&收益"
      icon={<Table2 size={18} />}
      rows={filtered}
      controls={<><SearchInput value={query} onChange={setQuery} placeholder="搜索品种/代码" /><select value={date} onChange={(e) => setDate(e.target.value)}>{dates.map((d) => <option key={d}>{d}</option>)}</select></>}
    />
  );
}

function TableView({ title, icon, rows, controls }: { title: string; icon: React.ReactNode; rows: Array<Record<string, unknown>>; controls: React.ReactNode }) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const columns = useMemo(() => Array.from(new Set(rows.flatMap((row) => Object.keys(row)))), [rows]);
  const visible = columns.filter((column) => !hidden.has(column));
  const toggle = (column: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(column)) next.delete(column);
      else next.add(column);
      return next;
    });
  };
  return (
    <div className="table-page">
      <div className="table-header">
        <div className="section-title">{icon}{title}</div>
        <div className="table-controls">{controls}</div>
      </div>
      <div className="column-toggles">
        {columns.map((column) => (
          <label key={column}><input type="checkbox" checked={!hidden.has(column)} onChange={() => toggle(column)} />{column}</label>
        ))}
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr>{visible.map((column) => <th key={column}>{column}</th>)}</tr></thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx}>{visible.map((column) => <td key={column}>{formatCell(row[column])}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatCell(value: unknown) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(Math.abs(value) < 1 ? 4 : 2);
  return String(value);
}

function SearchInput({ value, onChange, placeholder }: { value: string; onChange: (v: string) => void; placeholder: string }) {
  return <div className="search-box compact"><Search size={15} /><input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} /></div>;
}

function Logs({ logs, state }: { logs: string; state: RunState | null }) {
  const ref = useRef<HTMLPreElement | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  useEffect(() => {
    if (autoScroll && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [logs, autoScroll]);
  return (
    <div className="logs-page">
      <div className="table-header">
        <div className="section-title"><FileText size={18} />日志输出</div>
        <div className="table-controls">
          <label className="field inline"><input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} />自动滚动</label>
          <button className="secondary-button" onClick={() => navigator.clipboard.writeText(logs)}>复制</button>
        </div>
      </div>
      <pre ref={ref} className="log-box">{logs || (state ? '等待日志...' : '尚未运行')}</pre>
    </div>
  );
}

function Waiting({ state }: { state: RunState | null }) {
  return (
    <div className="empty-state">
      {state?.status === 'failed' ? <XCircle size={24} /> : state?.status === 'running' || state?.status === 'queued' ? <Loader2 className="spin" size={24} /> : <RotateCw size={24} />}
      {state?.status === 'failed' ? state.error || '运行失败' : state?.status === 'running' || state?.status === 'queued' ? '回测运行中，结果完成后显示。' : '运行回测后展示结果。'}
    </div>
  );
}

export default App;
