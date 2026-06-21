import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  BarChart3,
  CalendarClock,
  CheckCircle2,
  CircleDollarSign,
  ClipboardList,
  Download,
  FileText,
  Folder,
  FolderOpen,
  Home,
  ListFilter,
  Loader2,
  Play,
  RotateCw,
  Search,
  Settings,
  Table2,
  TerminalSquare,
  TrendingUp,
  XCircle
} from 'lucide-react';
import { api } from './api';
import type {
  EventFile,
  EventStudyResult,
  FactorDetail,
  FactorSummary,
  Metric,
  RunResult,
  RunState,
  TableMeta,
  TablePage
} from './types';

const PlotView = lazy(() => import('./PlotView'));

type Page = 'home' | 'detail' | 'eventstudy';
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

const EVENT_FALLBACK_PARAMS: Record<string, unknown> = {
  event_file: '',
  code: '000001.SZ',
  benchmark_code: '',
  metric: '收盘价(元)',
  table_name: 'daily_market',
  mode: 'flexible',
  window_before: 20,
  window_after: 20,
  holding_start_offset: 0,
  market_close_hour: 15,
  holding_days: '1,2,3,4,5',
  holding_months: '1,3,6,9,12'
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
        <nav className="top-tabs" aria-label="页面切换">
          <button className={`top-tab ${page === 'home' || page === 'detail' ? 'active' : ''}`} onClick={() => setPage('home')}>
            <BarChart3 size={15} />
            因子回测
          </button>
          <button className={`top-tab ${page === 'eventstudy' ? 'active' : ''}`} onClick={() => setPage('eventstudy')}>
            <CalendarClock size={15} />
            事件研究
          </button>
        </nav>
      </header>
      {error && <div className="global-error">{error}</div>}
      {page === 'home' && (
        <HomePage factors={factors} loading={loading} onOpen={openFactor} />
      )}
      {page === 'detail' && (
        <FactorPage factor={selected} detail={detail} onBack={() => setPage('home')} />
      )}
      {page === 'eventstudy' && (
        <EventStudyPage onBack={() => setPage('home')} />
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
          <p>从 `betalens-factor` 自动发现因子，按文件夹目录组织展示。</p>
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
        <FactorDirectory factors={filtered} allFactors={factors} onOpen={onOpen} />
      )}
    </main>
  );
}

function FactorDirectory({
  factors,
  allFactors,
  onOpen
}: {
  factors: FactorSummary[];
  allFactors: FactorSummary[];
  onOpen: (factor: FactorSummary) => void;
}) {
  const classes = useMemo(() => Array.from(new Set(allFactors.map((factor) => factor.factor_class))).sort(), [allFactors]);
  const [openDirs, setOpenDirs] = useState<Set<string>>(() => new Set(classes));

  useEffect(() => {
    setOpenDirs((prev) => {
      const next = new Set(prev);
      classes.forEach((cls) => next.add(cls));
      return next;
    });
  }, [classes]);

  const filteredByClass = useMemo(() => {
    const map = new Map<string, FactorSummary[]>();
    factors.forEach((factor) => {
      if (!map.has(factor.factor_class)) map.set(factor.factor_class, []);
      map.get(factor.factor_class)!.push(factor);
    });
    map.forEach((items) => items.sort((a, b) => a.name.localeCompare(b.name)));
    return map;
  }, [factors]);

  const toggleDir = (cls: string) => {
    setOpenDirs((prev) => {
      const next = new Set(prev);
      if (next.has(cls)) next.delete(cls);
      else next.add(cls);
      return next;
    });
  };

  if (!factors.length) {
    return <div className="empty-state">无匹配因子</div>;
  }

  return (
    <section className="factor-directory">
      {classes.map((cls) => {
        const visible = filteredByClass.get(cls) || [];
        if (!visible.length) return null;
        const isOpen = openDirs.has(cls);
        const total = allFactors.filter((factor) => factor.factor_class === cls).length;
        return (
          <div className="factor-folder" key={cls}>
            <button className="folder-header" onClick={() => toggleDir(cls)}>
              {isOpen ? <FolderOpen size={18} /> : <Folder size={18} />}
              <strong>{cls}</strong>
              <span>{visible.length === total ? `${total} 个因子` : `${visible.length} / ${total} 个因子`}</span>
            </button>
            {isOpen ? (
              <div className="factor-grid folder-factor-grid">
                {visible.map((factor) => (
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
              </div>
            ) : null}
          </div>
        );
      })}
    </section>
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
      //默认参数
      start_date: asString(detail.defaults.start_date, '2024-01-01'),
      end_date: asString(detail.defaults.end_date, '2025-12-31'),
      initial_amount: asNumber(detail.defaults.initial_amount, 100000000),
      rebal_freq: asString(detail.defaults.rebal_freq, 'W'),
      n_quantiles: asNumber(detail.defaults.n_quantiles, 80),
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
          {activeTab === 'trades' && <Trades runId={runId} result={result} />}
          {activeTab === 'positions' && <Positions runId={runId} result={result} />}
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

function EventStudyPage({ onBack }: { onBack: () => void }) {
  const [files, setFiles] = useState<EventFile[]>([]);
  const [loadingFiles, setLoadingFiles] = useState(true);
  const [params, setParams] = useState<Record<string, unknown>>(() => ({ ...EVENT_FALLBACK_PARAMS }));
  const [result, setResult] = useState<EventStudyResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.eventFiles()
      .then((payload) => {
        const items = payload.files || [];
        setFiles(items);
        setParams((prev) => ({
          ...EVENT_FALLBACK_PARAMS,
          ...(payload.defaults || {}),
          event_file: pickEventFile(payload.defaults, items, prev)
        }));
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoadingFiles(false));
  }, []);

  const selectedFile = files.find((file) => file.id === params.event_file);
  const update = (key: string, value: unknown) => setParams((prev) => ({ ...prev, [key]: value }));

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const data = await api.runEventStudy(params);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  };

  return (
    <main className="detail-page">
      <section className="run-header">
        <div className="title-block">
          <button className="ghost-button" onClick={onBack}>返回</button>
          <div>
            <h1>事件研究</h1>
            <p>扫描 betalens-factor/eventstudy 下的事件时点文件，分析事件窗口收益表现。</p>
          </div>
        </div>
        <div className="run-actions">
          <span className={`status-badge ${running ? 'running' : result ? 'completed' : 'idle'}`}>
            {running ? <Loader2 className="spin" size={16} /> : result ? <CheckCircle2 size={16} /> : <Activity size={16} />}
            {running ? '分析中' : result ? '分析完成' : '未运行'}
          </span>
          <button className="primary-button" onClick={run} disabled={running || !params.event_file}>
            {running ? <Loader2 className="spin" size={16} /> : <Play size={16} />}
            运行分析
          </button>
        </div>
      </section>

      <section className="settings-bar">
        <span>事件文件：</span>
        <strong>{selectedFile?.name || asString(params.event_file, '未选择')}</strong>
        <span>{selectedFile ? `${selectedFile.eventCount} 个事件` : ''}</span>
        <span>{asString(params.code)}</span>
        <span>{asString(params.window_before)} / {asString(params.window_after)} 天</span>
        {asString(params.benchmark_code) ? <span>基准 {asString(params.benchmark_code)}</span> : null}
      </section>

      {error && <div className="global-error">{error}</div>}

      <div className="detail-layout">
        <aside className="side-nav">
          <div className="parameter-panel">
            <h3><Settings size={16} />事件参数</h3>
            <label className="field">
              事件文件
              <select value={asString(params.event_file)} onChange={(event) => update('event_file', event.target.value)}>
                {loadingFiles ? <option>扫描中...</option> : null}
                {files.map((file) => (
                  <option key={file.id} value={file.id} disabled={Boolean(file.error)}>
                    {file.name} ({file.eventCount})
                  </option>
                ))}
              </select>
            </label>
            <LabeledInput label="标的代码" value={asString(params.code)} onChange={(v) => update('code', v)} />
            <LabeledInput label="基准代码" value={asString(params.benchmark_code)} onChange={(v) => update('benchmark_code', v)} />
            <LabeledInput label="价格指标" value={asString(params.metric)} onChange={(v) => update('metric', v)} />
            <LabeledInput label="数据表" value={asString(params.table_name)} onChange={(v) => update('table_name', v)} />
            <LabeledInput label="事件前窗口" type="number" value={asString(params.window_before)} onChange={(v) => update('window_before', Number(v))} />
            <LabeledInput label="事件后窗口" type="number" value={asString(params.window_after)} onChange={(v) => update('window_after', Number(v))} />
            <LabeledInput label="持有起点偏移" type="number" value={asString(params.holding_start_offset)} onChange={(v) => update('holding_start_offset', Number(v))} />
            <LabeledInput label="收盘小时" type="number" value={asString(params.market_close_hour)} onChange={(v) => update('market_close_hour', Number(v))} />
            <label className="field">
              累积模式
              <select value={asString(params.mode)} onChange={(event) => update('mode', event.target.value)}>
                <option value="flexible">事件窗口</option>
                <option value="fixed">固定持有期</option>
              </select>
            </label>
            {params.mode === 'fixed' ? (
              <>
                <LabeledInput label="固定天数" value={asString(params.holding_days)} onChange={(v) => update('holding_days', v)} />
                <LabeledInput label="固定月数" value={asString(params.holding_months)} onChange={(v) => update('holding_months', v)} />
              </>
            ) : null}
          </div>
        </aside>

        <section className="content-panel">
          <div className="view-stack">
            <EventFilePreview file={selectedFile} />
            {result ? <EventStudyResultView result={result} /> : <Waiting state={running ? { status: 'running' } as RunState : null} />}
          </div>
        </section>
      </div>
    </main>
  );
}

function pickEventFile(
  defaults: Record<string, unknown> | undefined,
  files: EventFile[],
  previous: Record<string, unknown>
) {
  const previousId = asString(previous.event_file);
  if (previousId && files.some((file) => file.id === previousId && !file.error)) return previousId;

  const defaultId = asString(defaults?.event_file);
  if (defaultId && files.some((file) => file.id === defaultId && !file.error)) return defaultId;

  return files.find((file) => !file.error)?.id || defaultId || previousId || '';
}

function EventFilePreview({ file }: { file?: EventFile }) {
  if (!file) return null;
  return (
    <div className="table-page">
      <div className="table-header">
        <div>
          <div className="section-title"><CalendarClock size={18} />事件文件</div>
          <div className="holding-subtitle">{file.path}</div>
        </div>
        <div className="event-file-stats">
          <span>{file.eventCount} 个事件</span>
          <span>{file.dateFrom} 至 {file.dateTo}</span>
        </div>
      </div>
      {file.error ? <div className="global-error">{file.error}</div> : <SimpleTable rows={file.sample} maxHeight={180} />}
    </div>
  );
}

function EventStudyResultView({ result }: { result: EventStudyResult }) {
  const daily = result.charts.dailyStats;
  const cumulative = result.charts.cumulativeStats;
  const matrix = result.charts.returnsMatrix;
  const cumulativeMatrix = result.charts.cumulativeReturnsMatrix || [];
  const summary = result.summary;
  const cumulativeEventSeries = useMemo(() => {
    const grouped = new Map<string, { day: number | string; value: number | null }[]>();
    cumulativeMatrix.forEach((row) => {
      const event = String(row.event ?? '');
      if (!event) return;
      const value = row.cumulativeReturn === null || row.cumulativeReturn === undefined
        ? null
        : Number(row.cumulativeReturn);
      if (!grouped.has(event)) grouped.set(event, []);
      grouped.get(event)!.push({
        day: row.day as number | string,
        value: Number.isFinite(value) ? value : null
      });
    });
    return Array.from(grouped.entries()).map(([event, rows]) => ({
      event,
      rows: rows.sort((a, b) => Number(a.day) - Number(b.day))
    }));
  }, [cumulativeMatrix]);

  return (
    <>
      <div className="section-title"><TrendingUp size={18} />分析结果</div>
      <div className="metrics-grid event-metrics-grid">
        <MetricTile label="事件数" value={summary.eventCount} />
        <MetricTile label="Day 0 平均收益" value={summary.day0Mean} percent />
        <MetricTile label="Day 0 t统计" value={summary.day0TStat} />
        <MetricTile label="Day 0 上涨概率" value={summary.day0PositiveProb} percent />
        <MetricTile label={`Day ${summary.finalDay ?? '-'} 累积收益`} value={summary.finalMean} percent />
        <MetricTile label="累积 t统计" value={summary.finalTStat} />
        <MetricTile label="累积上涨概率" value={summary.finalPositiveProb} percent />
      </div>
      <div className="chart-card">
        <Suspense fallback={<div className="chart-loading"><Loader2 className="spin" size={18} />加载图表...</div>}>
          <PlotView
            data={[
              {
                x: daily.map((row) => row.day),
                y: daily.map((row) => row.mean),
                type: 'bar',
                name: '平均收益率',
                marker: { color: daily.map((row) => asNumber(row.mean, 0) >= 0 ? '#6a9f42' : '#b94a48') },
                hovertemplate: 'Day %{x}<br>平均收益 %{y:.2%}<extra></extra>'
              }
            ]}
            layout={eventLayout('事件窗口平均收益率', 300)}
            config={{ displayModeBar: false, responsive: true }}
          />
        </Suspense>
      </div>
      <div className="chart-card">
        <Suspense fallback={<div className="chart-loading"><Loader2 className="spin" size={18} />加载图表...</div>}>
          <PlotView
            data={[
              {
                x: cumulative.map((row) => row.day),
                y: cumulative.map((row) => row.mean),
                type: 'scatter',
                mode: 'lines+markers',
                name: '平均累积收益',
                line: { color: '#2d66a8', width: 2 },
                hovertemplate: 'Day %{x}<br>累积收益 %{y:.2%}<extra></extra>'
              },
              {
                x: cumulative.map((row) => row.day),
                y: cumulative.map((row) => asNumber(row.mean, 0) + asNumber(row.std, 0)),
                type: 'scatter',
                mode: 'lines',
                name: '+1标准差',
                line: { color: '#9eb7d6', width: 1, dash: 'dot' },
                hoverinfo: 'skip'
              },
              {
                x: cumulative.map((row) => row.day),
                y: cumulative.map((row) => asNumber(row.mean, 0) - asNumber(row.std, 0)),
                type: 'scatter',
                mode: 'lines',
                name: '-1标准差',
                line: { color: '#9eb7d6', width: 1, dash: 'dot' },
                hoverinfo: 'skip'
              }
            ]}
            layout={eventLayout('平均累积收益率', 320)}
            config={{ displayModeBar: false, responsive: true }}
          />
        </Suspense>
      </div>
      {cumulativeEventSeries.length ? (
        <div className="chart-card">
          <Suspense fallback={<div className="chart-loading"><Loader2 className="spin" size={18} />加载图表...</div>}>
            <PlotView
              data={cumulativeEventSeries.map((series) => ({
                x: series.rows.map((row) => row.day),
                y: series.rows.map((row) => row.value),
                type: 'scatter',
                mode: 'lines',
                name: `事件 ${series.event}`,
                line: { width: 1.4 },
                opacity: 0.72,
                hovertemplate: `事件 ${series.event}<br>Day %{x}<br>累积收益 %{y:.2%}<extra></extra>`
              }))}
              layout={{
                ...eventLayout('每次事件前后累积收益', 360),
                showlegend: cumulativeEventSeries.length <= 12
              }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </Suspense>
        </div>
      ) : null}
      {matrix.length ? (
        <div className="chart-card">
          <Suspense fallback={<div className="chart-loading"><Loader2 className="spin" size={18} />加载图表...</div>}>
            <PlotView
              data={[
                {
                  x: matrix.map((row) => row.day),
                  y: matrix.map((row) => row.event),
                  z: matrix.map((row) => row.return),
                  type: 'scatter3d',
                  mode: 'markers',
                  marker: { size: 3, color: matrix.map((row) => row.return), colorscale: 'RdBu', reversescale: true, opacity: 0.75 },
                  name: '事件收益点'
                }
              ]}
              layout={{
                title: { text: '三维事件收益矩阵', font: { size: 15 } },
                height: 420,
                margin: { l: 0, r: 0, t: 42, b: 0 },
                scene: {
                  xaxis: { title: { text: '相对日' } },
                  yaxis: { title: { text: '事件' } },
                  zaxis: { title: { text: '收益率' }, tickformat: '.1%' }
                },
                paper_bgcolor: '#ffffff'
              }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </Suspense>
        </div>
      ) : null}
      <div className="table-page">
        <div className="section-title"><Table2 size={18} />日度统计</div>
        <SimpleTable rows={result.tables.dailyStats} maxHeight={360} />
      </div>
      <div className="table-page">
        <div className="section-title"><Table2 size={18} />累积统计</div>
        <SimpleTable rows={result.tables.cumulativeStats} maxHeight={360} />
      </div>
    </>
  );
}

function MetricTile({ label, value, percent = false }: { label: string; value: unknown; percent?: boolean }) {
  let display = '-';
  if (value !== null && value !== undefined && value !== '') {
    const num = Number(value);
    display = Number.isFinite(num)
      ? percent ? `${(num * 100).toFixed(2)}%` : Math.abs(num) >= 100 ? num.toFixed(2) : num.toFixed(3)
      : String(value);
  }
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong className={Number(value) < 0 ? 'negative' : ''}>{display}</strong>
    </div>
  );
}

function eventLayout(title: string, height: number) {
  return {
    ...baseLayout(title, height, false),
    shapes: [
      { type: 'line' as const, x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper' as const, line: { color: '#b94a48', width: 1, dash: 'dash' as const } },
      { type: 'line' as const, x0: 0, x1: 1, xref: 'paper' as const, y0: 0, y1: 0, line: { color: '#87909a', width: 1, dash: 'dot' as const } }
    ],
    yaxis: { gridcolor: '#d8dde3', zerolinecolor: '#87909a', tickformat: '.1%' }
  };
}

function SimpleTable({ rows, maxHeight = 260 }: { rows: Array<Record<string, unknown>>; maxHeight?: number }) {
  const columns = useMemo(() => Array.from(new Set(rows.flatMap((row) => Object.keys(row)))), [rows]);
  if (!rows.length) return <div className="table-empty">无数据</div>;
  return (
    <div className="table-wrap" style={{ maxHeight }}>
      <table>
        <thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>{columns.map((column) => <td key={column}>{formatCell(row[column])}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Overview({ result, state }: { result: RunResult | null; state: RunState | null }) {
  if (!result) return <Waiting state={state} />;
  const nav = result.charts.nav;
  const pnl = result.charts.dailyPnl;
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
        <Suspense fallback={<div className="chart-loading"><Loader2 className="spin" size={18} />加载图表...</div>}>
          <PlotView
            data={[
              { x: nav.map((p) => p.date), y: nav.map((p) => p.nav), type: 'scatter', mode: 'lines', name: '策略净值', line: { color: '#2d66a8', width: 2 } }
            ]}
            layout={baseLayout('收益净值曲线', 360, true)}
            config={{ displayModeBar: false, responsive: true }}
          />
        </Suspense>
      </div>
      <div className="chart-card">
        <Suspense fallback={<div className="chart-loading"><Loader2 className="spin" size={18} />加载图表...</div>}>
          <PlotView
            data={[
              { x: pnl.map((p) => p.date), y: pnl.map((p) => p.pnl), type: 'bar', name: '每日盈亏', marker: { color: pnl.map((p) => Number(p.pnl) >= 0 ? '#6a9f42' : '#8061a8') } }
            ]}
            layout={baseLayout('每日盈亏', 260, false)}
            config={{ displayModeBar: false, responsive: true }}
          />
        </Suspense>
      </div>
      <RebalanceHoldings records={result.charts.rebalanceHoldings || []} />
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

function RebalanceHoldings({ records }: { records: Array<Record<string, number | string | null>> }) {
  const [selectedDate, setSelectedDate] = useState('');
  const [query, setQuery] = useState('');

  const grouped = useMemo(() => {
    const map = new Map<string, Array<Record<string, number | string | null>>>();
    records.forEach((record) => {
      const date = asString(record.date);
      if (!date) return;
      if (!map.has(date)) map.set(date, []);
      map.get(date)!.push(record);
    });
    map.forEach((rows) => {
      rows.sort((a, b) => asNumber(a.rank, 0) - asNumber(b.rank, 0));
    });
    return map;
  }, [records]);

  const dates = useMemo(() => Array.from(grouped.keys()).sort(), [grouped]);
  const activeDate = selectedDate && grouped.has(selectedDate) ? selectedDate : dates[dates.length - 1] || '';
  const rows = (grouped.get(activeDate) || []).filter((row) => {
    const text = `${row.code ?? ''} ${row.name ?? ''}`.toLowerCase();
    return text.includes(query.toLowerCase());
  });
  const totalWeight = rows.reduce((sum, row) => sum + Math.abs(asNumber(row.weight, 0)), 0);

  return (
    <div className="holding-panel">
      <div className="table-header">
        <div>
          <div className="section-title"><Table2 size={18} />调仓日三维持仓列表</div>
          <div className="holding-subtitle">时间轴 / 持股代码与中文名称 / 因子值</div>
        </div>
        <div className="table-controls">
          <SearchInput value={query} onChange={setQuery} placeholder="搜索代码/名称" />
        </div>
      </div>

      {!records.length ? (
        <div className="table-empty">无调仓持仓数据</div>
      ) : (
        <div className="holding-layout">
          <div className="rebalance-timeline">
            {dates.map((date) => {
              const count = grouped.get(date)?.length || 0;
              return (
                <button
                  key={date}
                  className={`timeline-item ${date === activeDate ? 'active' : ''}`}
                  onClick={() => setSelectedDate(date)}
                >
                  <span>{date}</span>
                  <strong>{count}</strong>
                </button>
              );
            })}
          </div>
          <div className="holding-detail">
            <div className="holding-summary">
              <strong>{activeDate}</strong>
              <span>{rows.length} 只持仓</span>
              <span>权重合计 {formatPercent(totalWeight)}</span>
            </div>
            <div className="table-wrap holding-table-wrap">
              <table className="holding-table">
                <thead>
                  <tr>
                    <th>序号</th>
                    <th>股票代码</th>
                    <th>中文名称</th>
                    <th>权重</th>
                    <th>因子值</th>
                    <th>分组</th>
                    <th>信号日</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={`${row.date}-${row.code}`}>
                      <td>{formatCell(row.rank)}</td>
                      <td>{asString(row.code)}</td>
                      <td>{asString(row.name)}</td>
                      <td>{formatPercent(asNumber(row.weight, 0))}</td>
                      <td>{formatFactor(row.factorValue)}</td>
                      <td>{formatCell(row.group)}</td>
                      <td>{asString(row.signalDate, '-')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!rows.length && <div className="table-empty">无匹配持仓</div>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function formatPercent(value: number) {
  if (!Number.isFinite(value)) return '-';
  return `${(value * 100).toFixed(2)}%`;
}

function formatFactor(value: unknown) {
  if (value === null || value === undefined || value === '') return '-';
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  if (Math.abs(num) >= 100) return num.toFixed(2);
  if (Math.abs(num) >= 1) return num.toFixed(4);
  return num.toFixed(6);
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

function Trades({ runId, result }: { runId: string | null; result: RunResult | null }) {
  const [direction, setDirection] = useState('全部');
  if (!result) return <Waiting state={null} />;
  return (
    <ResultTable
      runId={runId}
      kind="trades"
      title="交易详情"
      icon={<ClipboardList size={18} />}
      meta={result.tables.trades}
      filters={direction === '全部' ? {} : { direction }}
      extraControls={
        <select value={direction} onChange={(e) => setDirection(e.target.value)}>
          <option>全部</option>
          <option value="buy">buy</option>
          <option value="sell">sell</option>
        </select>
      }
    />
  );
}

function Positions({ runId, result }: { runId: string | null; result: RunResult | null }) {
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  if (!result) return <Waiting state={null} />;
  return (
    <ResultTable
      runId={runId}
      kind="positions"
      title="每日持仓&收益"
      icon={<Table2 size={18} />}
      meta={result.tables.positions}
      dateFrom={dateFrom}
      dateTo={dateTo}
      extraControls={
        <>
          <LabeledInlineInput label="开始" type="date" value={dateFrom} onChange={setDateFrom} />
          <LabeledInlineInput label="结束" type="date" value={dateTo} onChange={setDateTo} />
        </>
      }
    />
  );
}

function ResultTable({
  runId,
  kind,
  title,
  icon,
  meta,
  filters = {},
  dateFrom,
  dateTo,
  extraControls
}: {
  runId: string | null;
  kind: 'trades' | 'positions';
  title: string;
  icon: React.ReactNode;
  meta: TableMeta;
  filters?: Record<string, string>;
  dateFrom?: string;
  dateTo?: string;
  extraControls?: React.ReactNode;
}) {
  const [page, setPage] = useState(1);
  const [size] = useState(50);
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [data, setData] = useState<TablePage | null>(null);
  const [loading, setLoading] = useState(false);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [tableError, setTableError] = useState<string | null>(null);

  const filterKey = JSON.stringify(filters);
  const dateKey = `${dateFrom ?? ''}:${dateTo ?? ''}`;

  // 搜索框防抖,避免每次按键都打后端
  useEffect(() => {
    const handle = window.setTimeout(() => setDebouncedQuery(query), 300);
    return () => window.clearTimeout(handle);
  }, [query]);

  // 过滤条件或搜索词变化时回到第一页
  useEffect(() => {
    setPage(1);
  }, [debouncedQuery, filterKey, dateKey]);

  const load = () => {
    if (!runId) return;
    setLoading(true);
    setTableError(null);
    api
      .table(runId, kind, { page, size, query: debouncedQuery, filters, dateFrom, dateTo })
      .then(setData)
      .catch((err) => setTableError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  };

  useEffect(load, [runId, kind, page, size, debouncedQuery, filterKey, dateKey]);

  const columns = meta.columns;
  const visible = columns.filter((column) => !hidden.has(column));
  const toggle = (column: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(column)) next.delete(column);
      else next.add(column);
      return next;
    });

  const rows = data?.rows ?? [];
  const total = data?.total ?? 0;
  const pages = data?.pages ?? 0;

  return (
    <div className="table-page">
      <div className="table-header">
        <div className="section-title">{icon}{title}</div>
        <div className="table-controls">
          <SearchInput value={query} onChange={setQuery} placeholder="搜索代码/字段" />
          {extraControls}
          <button className="secondary-button" onClick={load} title="刷新">
            <RotateCw size={15} className={loading ? 'spin' : ''} />刷新
          </button>
        </div>
      </div>
      <div className="column-toggles">
        {columns.map((column) => (
          <label key={column}><input type="checkbox" checked={!hidden.has(column)} onChange={() => toggle(column)} />{column}</label>
        ))}
      </div>
      {tableError && <div className="global-error">{tableError}</div>}
      <div className="table-wrap">
        <table>
          <thead><tr>{visible.map((column) => <th key={column}>{column}</th>)}</tr></thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx}>{visible.map((column) => <td key={column}>{formatCell(row[column])}</td>)}</tr>
            ))}
          </tbody>
        </table>
        {loading && <div className="table-loading"><Loader2 className="spin" size={18} />加载中...</div>}
        {!loading && !rows.length && <div className="table-empty">无数据</div>}
      </div>
      <Pagination page={page} pages={pages} total={total} size={size} loading={loading} onPage={setPage} />
    </div>
  );
}

function Pagination({
  page,
  pages,
  total,
  size,
  loading,
  onPage
}: {
  page: number;
  pages: number;
  total: number;
  size: number;
  loading: boolean;
  onPage: (page: number) => void;
}) {
  const from = total ? (page - 1) * size + 1 : 0;
  const to = Math.min(page * size, total);
  return (
    <div className="pagination">
      <span className="page-info">共 {total.toLocaleString()} 行，显示 {from}-{to}</span>
      <div className="page-controls">
        <button className="ghost-button" disabled={loading || page <= 1} onClick={() => onPage(1)}>首页</button>
        <button className="ghost-button" disabled={loading || page <= 1} onClick={() => onPage(page - 1)}>上一页</button>
        <span className="page-current">{page} / {pages || 1}</span>
        <button className="ghost-button" disabled={loading || page >= pages} onClick={() => onPage(page + 1)}>下一页</button>
        <button className="ghost-button" disabled={loading || page >= pages} onClick={() => onPage(pages)}>末页</button>
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

function LabeledInlineInput({
  label,
  value,
  onChange,
  type = 'text'
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
}) {
  return (
    <label className="inline-input">
      <span>{label}</span>
      <input type={type} value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
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
