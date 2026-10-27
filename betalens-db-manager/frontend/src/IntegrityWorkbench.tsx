import React, {useEffect, useRef, useState} from 'react';

type Row = Record<string, any>;
type Api = (path: string, body?: any) => Promise<any>;
type CheckRequest = {
  targets: string[]; modules: string[]; security_codes: string[];
  start_date: string; end_date: string; exchanges: string[];
  schemes: string[]; levels: number[]; check_times: string[];
  source_id?: string;
  requests_per_minute?: number;
};
const modules = [['calendar', '交易日历'], ['industry', '股票历史行业'], ['suspension', '股票停牌区间']];
const targetNames: Row = {postgres: 'PostgreSQL', clickhouse: 'ClickHouse'};
const labels: Row = {
  target: '目标库', module: '业务模块', status: '状态', reason: '原因', reasons: '原因',
  security_code: '股票代码', exchange: '交易所', date: '日期', trade_date: '业务日期',
  scheme_code: '分类体系', level: '层级', industry_code: '行业代码', industry_name: '行业名称',
  valid_from: '有效起日', valid_to: '有效止日', is_open: '是否开市', previous_trade_date: '上一交易日',
  import_task_id: '任务编号', task_id: '任务编号', attempt_id: '执行尝试', revision_id: '修订编号',
  scope_id: '范围编号', ingestion_seq: '入库序号', ingested_at: '实际入库时间',
  published_at: '发布时间', available_at: '实际可得时间', downloaded_at: '下载时间',
  event_date: '业务发生日期', source_name: '来源', source_endpoint: '源接口',
  availability_basis: '可得时间依据', confidence: '可信程度', completeness: '完整性',
  rows: '行数', records: '记录', proof: '范围凭证', evidence: '依据', dependencies: '依赖',
  before: '写入前', after: '拟写入', column: '字段', http_status: 'HTTP 状态', duration: '耗时',
};
function asRows(value: any): Row[] {
  if (Array.isArray(value)) return value.map(item => item !== null && typeof item === 'object' ? item : {value: item});
  if (value !== null && typeof value === 'object') return Object.entries(value).map(([field, value]) => ({field, value}));
  return value === undefined || value === null ? [] : [{value}];
}
function Cell({value, field}: {value: any; field?: string}) {
  if (value === undefined || value === null || value === '') return <span className="muted">—</span>;
  if (typeof value === 'object') return <details><summary>{Array.isArray(value) ? `${value.length} 条明细` : '查看明细'}</summary><DataTable rows={asRows(value)}/></details>;
  if (field === 'raw_record' && typeof value === 'string') {
    try {const parsed = JSON.parse(value); if (parsed !== null && typeof parsed === 'object') return <details><summary>源记录字段</summary><DataTable rows={asRows(parsed)}/></details>;} catch {/* Preserve malformed source text for inspection. */}
  }
  if (typeof value === 'boolean') return <>{value ? '是' : '否'}</>;
  if (typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$/.test(value)) return <time title={value}>{new Date(value).toLocaleString('zh-CN', {timeZone: 'Asia/Shanghai', hour12: false})}</time>;
  return <>{String(value)}</>;
}
function DataTable({rows = [], empty = '暂无记录', testId}: {rows?: Row[]; empty?: string; testId?: string}) {
  if (!rows.length) return <p className="muted" data-testid={testId}>{empty}</p>;
  const keys = [...new Set(rows.flatMap(row => Object.keys(row)))];
  return <div className="scroll reference-table" data-testid={testId}><table><thead><tr>{keys.map(key => <th key={key} title={key}>{labels[key] || key}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index} data-business-date={row.trade_date || row.date} data-module={row.module} data-security-code={row.security_code}>{keys.map(key => <td key={key}><Cell value={row[key]} field={key}/></td>)}</tr>)}</tbody></table></div>;
}
function Options({label, options, values, onChange}: {label: string; options: string[][]; values: string[]; onChange: (value: string[]) => void}) {
  return <fieldset className="reference-options"><legend>{label}</legend><div className="checks">{options.map(([key, name]) => <label key={key}><input type="checkbox" checked={values.includes(key)} onChange={event => onChange(event.target.checked ? [...values, key] : values.filter(value => value !== key))}/>{name}</label>)}</div></fieldset>;
}
function todayOffset(days: number) {
  const date = new Date(Date.now() + days * 86400000);
  const parts = new Intl.DateTimeFormat('en-CA', {timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit'}).formatToParts(date);
  return ['year', 'month', 'day'].map(key => parts.find(part => part.type === key)!.value).join('-');
}
const split = (value: string) => [...new Set(value.split(/[\s,，]+/).map(item => item.trim()).filter(Boolean))];
const taskRows = (value: any): Row[] => Array.isArray(value) ? value : value?.tasks || value?.rows || [];
function requestOf(job?: Row) {return job?.request || job?.plan?.request || {};}
function auditOf(job?: Row) {return job?.preview || job?.audit || {};}
const busyStates = new Set(['queued', 'running', 'downloading', 'committing', 'rolling_back', 'publishing']);
const finalStates = new Set(['completed', 'partial', 'partially_completed', 'rolled_back', 'revoked', 'failed', 'interrupted']);

export default function IntegrityWorkbench({api}: {api: Api}) {
  const [targets, setTargets] = useState(['postgres', 'clickhouse']);
  const [sourceId, setSourceId] = useState('tushare');
  const [sources, setSources] = useState<Row[]>([]);
  const [selectedModules, setModules] = useState(['calendar', 'industry', 'suspension']);
  const [codes, setCodes] = useState('000001.SZ');
  const [start, setStart] = useState(todayOffset(-30));
  const [end, setEnd] = useState(todayOffset(-1));
  const [exchanges, setExchanges] = useState(['SSE', 'SZSE', 'BSE']);
  const [schemes, setSchemes] = useState(['SW2021', 'CITICS']);
  const [levels, setLevels] = useState(['1', '2', '3']);
  const [times, setTimes] = useState('10:00:00');
  const [checking, setChecking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [report, setReport] = useState<Row>();
  const [checkedRequest, setCheckedRequest] = useState<CheckRequest>();
  const [draft, setDraft] = useState<CheckRequest>();
  const [draftHint, setDraftHint] = useState('');
  const [tasks, setTasks] = useState<Row[]>([]);
  const [job, setJob] = useState<Row>();
  const [acceptPartial, setAcceptPartial] = useState(false);
  const [tables, setTables] = useState<Row[]>([]);
  const [schema, setSchema] = useState<Row>();
  const [schemaResult, setSchemaResult] = useState<Row>();
  const [versionTarget, setVersionTarget] = useState('postgres');
  const [allVersions, setAllVersions] = useState(false);
  const [versions, setVersions] = useState<Row>();
  const [versionSelected, setVersionSelected] = useState('');
  const [requestRpm, setRequestRpm] = useState(40);
  const [baselineTasks, setBaselineTasks] = useState('');
  const taskGeneration = useRef(0);
  const selectedTaskId = useRef<string | undefined>(undefined);
  const jobStatus = useRef('');
  const current = (): CheckRequest => ({targets, modules: selectedModules, security_codes: split(codes), start_date: start, end_date: end, exchanges, schemes, levels: levels.map(Number), check_times: split(times), source_id: sourceId, requests_per_minute: requestRpm});

  async function run(action: () => Promise<void>) {
    setBusy(true); setError('');
    try {await action();} catch (error: any) {setError(error.message || '请求失败');} finally {setBusy(false);}
  }
  async function listTasks() {setTasks(taskRows(await api('/reference/tasks')));}
  function selectTask(value?: Row) {
    if (selectedTaskId.current !== value?.id) taskGeneration.current += 1;
    selectedTaskId.current = value?.id;
    setJob(value); setAcceptPartial(false); jobStatus.current = value?.status || '';
    const url = new URL(location.href);
    if (value?.id) url.searchParams.set('task', value.id); else url.searchParams.delete('task');
    history.replaceState(null, '', url.pathname + url.search);
  }
  useEffect(() => {
    let active = true;
    const task = new URLSearchParams(location.search).get('task');
    Promise.all([api('/reference/tasks'), api('/reference/tables'), task ? api('/reference/tasks/' + encodeURIComponent(task)) : Promise.resolve(null), api('/sources'), api('/config')]).then(([jobs, catalog, selected, sourceList, config]) => {
      if (!active) return;
      setTasks(taskRows(jobs)); setTables(Array.isArray(catalog) ? catalog : catalog?.tables || catalog?.rows || []);
      setSources(Array.isArray(sourceList) ? sourceList : []);
      setRequestRpm(Number(config.requests_per_minute) || 40);
      if (selected) selectTask(selected);
    }).catch(error => {if (active) setError(error.message);});
    return () => {active = false; taskGeneration.current += 1;};
  }, []);
  useEffect(() => {
    if (!job?.id) return;
    const id = job.id; const generation = taskGeneration.current;
    let active = true;
    const refresh = async () => {
      try {
        const latest = await api('/reference/tasks/' + encodeURIComponent(id));
        if (!active || generation !== taskGeneration.current) return;
        setJob(latest);
        if (latest.status !== jobStatus.current) {
          jobStatus.current = latest.status;
          await listTasks();
          if (finalStates.has(latest.status) && checkedRequest) {
            const next = await api('/integrity/check', checkedRequest);
            if (active && generation === taskGeneration.current) setReport(next);
          }
        }
      } catch (error: any) {if (active) setError('任务状态读取失败：' + error.message);}
    };
    const timer = setInterval(refresh, 2500);
    return () => {active = false; clearInterval(timer);};
  }, [job?.id, checkedRequest]);

  async function check() {
    setChecking(true); setError('');
    try {const request = current(); setReport(await api('/integrity/check', request)); setCheckedRequest(request);}
    catch (error: any) {setError(error.message);} finally {setChecking(false);}
  }
  function prepareDraft(row?: Row, target?: string) {
    const request = checkedRequest || current();
    const date = row?.date || row?.trade_date || row?.business_date;
    const module = row?.module && modules.some(([value]) => value === row.module) ? row.module : undefined;
    const scheme = row?.scheme_code || row?.scheme;
    setDraft({...request,
      targets: target ? [target] : [...request.targets],
      modules: module ? [module] : [...request.modules],
      security_codes: row?.security_code ? [row.security_code] : [...request.security_codes],
      exchanges: row?.exchange ? [row.exchange] : [...request.exchanges],
      schemes: module === 'industry' && scheme ? [scheme] : [...request.schemes],
      levels: module === 'industry' && row?.level ? [Number(row.level)] : [...request.levels],
      start_date: module === 'industry' || !date ? request.start_date : date,
      end_date: module === 'industry' || !date ? request.end_date : date,
    });
    setDraftHint(module === 'industry' ? '行业按股票 × 体系 × 层级拉取完整历史快照；检查日期用于定位有效区间。' : '请核对下方实际请求范围。创建任务只记录参数，开始拉取后才会请求来源。');
    setNotice('已生成待核对的请求参数，尚未开始拉取。');
  }
  async function action(operation: string, body: Row = {}) {
    if (!job?.id) return;
    const id = job.id;
    const result = await api('/reference/tasks/' + encodeURIComponent(id) + '/' + operation, body);
    if (operation === 'retry') selectTask(result.task || result.job || result);
    else setJob(await api('/reference/tasks/' + encodeURIComponent(id)));
    if (operation === 'preview') setAcceptPartial(false);
    await listTasks();
    setNotice(operation === 'retry' ? '已创建关联的新任务；需重新拉取、审计和确认。' : operation === 'preview' ? '审计已更新，请核对缓存、版本头和差异后确认入库。' : operation === 'download' ? '拉取操作已登记，数据库尚未写入；请查看实时日志。' : operation === 'rollback' ? '撤销发布操作已登记；历史业务行和审计保留。' : '写入操作已登记，分别核验目标库及业务模块后发布。');
    if (['commit', 'rollback'].includes(operation) && checkedRequest && result.status && finalStates.has(result.status)) setReport(await api('/integrity/check', checkedRequest));
  }
  const preview = auditOf(job);
  const targetResults = report?.results || {};
  const selectedVersionRows = asRows(versions?.rows).filter(row => !versionSelected || row.revision_id === versionSelected);
  const locked = busy || (job && busyStates.has(job.status));
  const rejected = Number(preview.rejected || preview.rejected_count || 0);
  const previewTables = Array.isArray(preview.tables) ? preview.tables : [];
  const outcome = job?.result || {};
  return <div className="reference-workbench">
    {error && <div role="alert" className="error">{error}</div>}
    {notice && <div role="status" className="notice">{notice}</div>}
    <section>
      <h2>1 · 检查业务参考表完整性</h2>
      <p>分别依据目标库中的可信发布和请求范围凭证检查。未请求、未发布、缺失及含义不明会返回原因；其他股票的记录不能证明当前股票完整。</p>
      <Options label="检查目标库" options={Object.entries(targetNames)} values={targets} onChange={value => {setTargets(value); setSchema(undefined);}}/>
      <Options label="业务模块" options={modules} values={selectedModules} onChange={setModules}/>
      <div className="grid">
        <label>参考表更新来源<select aria-label="参考表更新来源" value={sourceId} onChange={event => setSourceId(event.target.value)}><option value="tushare">Tushare</option>{sources.filter(source => source.source_id !== 'tushare' && Object.keys(source.endpoints || {}).length).map(source => <option key={source.source_id} value={source.source_id} disabled={!source.configured}>{source.label || source.source_id} · {source.configured ? '已配置' : '未配置'}</option>)}</select><small>检查只读取已发布版本；来源选择用于据此登记的更新任务。</small></label>
        <label>检查起始日期<input aria-label="完整性起始日期" type="date" value={start} onChange={event => setStart(event.target.value)}/></label>
        <label>检查结束日期<input aria-label="完整性结束日期" type="date" value={end} onChange={event => setEnd(event.target.value)}/></label>
        <label>股票代码<input aria-label="完整性股票代码" value={codes} onChange={event => setCodes(event.target.value)} placeholder="000001.SZ，600000.SH"/><small>按逗号或空格分隔，行业及停牌逐股票检查。</small></label>
        {selectedModules.includes('suspension') && <label>北京时间检查时刻<input aria-label="停牌检查时刻" value={times} onChange={event => setTimes(event.target.value)} placeholder="10:00:00，14:00:00"/><small>仅判断指定时刻是否处于停牌区间。</small></label>}
      </div>
      {selectedModules.includes('calendar') && <Options label="日历交易所" options={['SSE', 'SZSE', 'BSE'].map(value => [value, value])} values={exchanges} onChange={setExchanges}/>}
      {selectedModules.includes('industry') && <><Options label="行业分类体系" options={[[ 'SW2021', '申万 2021'], ['CITICS', '中信']]} values={schemes} onChange={setSchemes}/><Options label="行业层级" options={['1', '2', '3'].map(value => [value, value + ' 级'])} values={levels} onChange={setLevels}/></>}
      <div className="actions"><button className="primary" disabled={checking || busy || !targets.length || !selectedModules.length || !start || !end || start > end} onClick={check}>{checking ? '正在检查…' : '检查业务完整性'}</button><button disabled={busy || !targets.length || !selectedModules.length} onClick={() => prepareDraft()}>按当前范围指定更新请求</button></div>
      {report && <div data-testid="integrity-results">
        {Object.entries(targetResults).map(([target, value]: [string, any]) => <div className="reference-target-result" key={target}>
          <h3>{targetNames[target] || target} · 完整性结果</h3>
          <DataTable rows={asRows(value.summary)} empty="暂无检查摘要"/>
          <DataTable rows={value.rows || value.data || []} testId={'integrity-rows-' + target}/>
          {value.error && <p className="error">{value.error}</p>}
          <div className="reference-gap-actions">{(value.rows || []).filter((row: Row) => row.status !== 'complete' && row.completeness !== 'complete' && row.status !== 'known').slice(0, 60).map((row: Row, index: number) => <button key={index} onClick={() => prepareDraft(row, target)}>按此范围准备任务 · {row.module || '参考表'} · {row.security_code || row.exchange || ''} · {row.date || row.trade_date || ''}</button>)}</div>
        </div>)}
        <p className="muted">完整无事件响应仅表示“未报告停牌”；缺少凭证或日内边界不明不会按正常交易处理。实际可得时间未知时保留空值。</p>
      </div>}
    </section>

    {draft && <section data-testid="reference-request-draft">
      <h2>2 · 核对并登记更新任务</h2><p>{draftHint}</p>
      <DataTable rows={[draft]}/>
      <div className="grid"><label>请求起始日期<input aria-label="参考表请求起始日期" type="date" value={draft.start_date} onChange={event => setDraft({...draft, start_date: event.target.value})}/></label><label>请求结束日期<input aria-label="参考表请求结束日期" type="date" value={draft.end_date} onChange={event => setDraft({...draft, end_date: event.target.value})}/></label><label>请求股票代码<input aria-label="参考表请求股票代码" value={draft.security_codes.join(', ')} onChange={event => setDraft({...draft, security_codes: split(event.target.value)})}/></label></div>
      <Options label="请求目标库" options={Object.entries(targetNames)} values={draft.targets} onChange={targets => setDraft({...draft, targets})}/>
      <Options label="请求业务模块" options={modules} values={draft.modules} onChange={modules => setDraft({...draft, modules})}/>
      <label>请求来源<select aria-label="参考表请求来源" value={draft.source_id || 'tushare'} onChange={event => setDraft({...draft, source_id: event.target.value})}><option value="tushare">Tushare</option>{sources.filter(source => source.source_id !== 'tushare' && Object.keys(source.endpoints || {}).length).map(source => <option key={source.source_id} value={source.source_id} disabled={!source.configured}>{source.label || source.source_id}</option>)}</select></label>
      {(draft.source_id || 'tushare') === 'tushare' && <label>Tushare 每分钟请求上限<input aria-label="参考表每分钟请求次数" type="number" min="1" max="10000" value={draft.requests_per_minute || requestRpm} onChange={event => setDraft({...draft, requests_per_minute: Number(event.target.value)})}/><small>同时遵守接口自身限频；按分页和分区完整获取，不以短页推断结束。</small></label>}
      <p>新版本独立生成；入库后先核验再发布，不从旧版本静默补值。行业查询需要完整历史快照。</p>
      <button className="primary" disabled={busy || !draft.targets.length || !draft.modules.length || draft.start_date > draft.end_date} onClick={() => run(async () => {const result = await api('/reference/tasks', draft); selectTask(result.task || result.job || result); await listTasks(); setDraft(undefined); setNotice('任务已持久化，仅登记请求参数；点击“开始拉取参考数据”才会访问来源。');})}>登记参考表更新任务</button>
      <details><summary>使用已有归档准备基线</summary><p>只复用来源身份、字段合同、请求参数及完整分页证据均匹配的归档；旧业务行和“任务已完成”状态不会成为可信依据。缺少证据的范围标为待重新获取，仍需缓存审计和确认。</p><label>源任务编号<input aria-label="参考表基线源任务编号" value={baselineTasks} onChange={event => setBaselineTasks(event.target.value)} placeholder="任务编号，以逗号或空格分隔"/></label><button disabled={busy || !split(baselineTasks).length || !draft.targets.length || !draft.modules.length} onClick={() => run(async () => {const result = await api('/reference/baselines', {request: draft, source_task_ids: split(baselineTasks)}); selectTask(result); await listTasks(); setDraft(undefined); setNotice('已准备基线归档，未写入数据库；请核对归档证明并审计，缺少证据的范围需显式拉取。');})}>准备可审计基线任务</button></details>
      <button onClick={() => setDraft(undefined)}>取消待登记请求</button>
    </section>}

    <section>
      <div className="heading"><h2>3 · 任务、缓存审计与可信发布</h2><button disabled={busy} onClick={() => run(listTasks)}>刷新参考表任务</button></div>
      <div className="reference-job-list">{tasks.map(task => <button className={job?.id === task.id ? 'selected' : ''} key={task.id} onClick={() => run(async () => selectTask(await api('/reference/tasks/' + encodeURIComponent(task.id))))}><code>{task.id}</code><span>{task.status}</span></button>)}</div>
      {!tasks.length && <p className="muted">尚无参考表任务。检查缺口后登记；不会自动启动历史补全。</p>}
      {job && <div data-testid="reference-task">
        <h3>任务 {job.id}</h3><DataTable rows={[{id: job.id, status: job.status, parent_task_id: job.parent_task_id, ...requestOf(job)}]}/>
        <details><summary>固定的实际 API 请求、来源和映射</summary><DataTable rows={asRows(job.requests || job.plan?.requests)}/><DataTable rows={asRows(job.mapping_snapshots || job.plan?.mapping_snapshots)}/></details>
        {job.plan?.baseline_report && <><h3>旧归档基线核验与待重新获取范围</h3><DataTable rows={asRows(job.plan.baseline_report)}/></>}
        {job.error && <div className="error">{job.error}</div>}
        <div className="actions">
          <button disabled={locked} onClick={() => run(() => action('download'))}>开始拉取参考数据</button>
          <button disabled={locked} onClick={() => run(() => action('preview'))}>检查参考数据缓存与差异</button>
          <button disabled={locked} onClick={() => run(() => action('retry'))}>建立关联的新版本任务</button>
        </div>
        <h3>执行尝试与请求日志</h3><DataTable rows={asRows(job.attempts || job.execution_attempts)} empty="尚无执行尝试"/><DataTable rows={asRows(job.logs)} testId="reference-task-logs" empty="开始拉取后显示接口、脱敏参数、HTTP 状态、耗时、分页和重试记录。"/>
        {job.preview && <div className="reference-audit" data-testid="reference-audit">
          <h3>请求完整性、依赖与版本头</h3><DataTable rows={asRows(preview.source_report || preview.checks || preview.integrity)}/><DataTable rows={asRows(preview.scopes)}/><DataTable rows={asRows(preview.heads || preview.version_heads || preview.target_heads)}/><DataTable rows={asRows(preview.dependencies)}/>
          <h3>原始响应缓存</h3><DataTable rows={asRows(preview.raw_sample || preview.raw_rows || preview.source_rows)} empty="本次审计未提供原始样本；请查看来源检查记录。"/>
          <h3>字段转换与标准化记录</h3><DataTable rows={asRows(preview.field_validation || preview.validation)}/><DataTable rows={asRows(preview.normalized_sample || preview.normalized_rows || preview.rows)}/>
          <h3>追加版本、替代范围及 before / after</h3><DataTable rows={asRows(preview.changes || preview.change_sample)}/>
          {previewTables.map((table: Row, index: number) => <details key={index}><summary>{table.target} · {table.table || table.module}</summary><DataTable rows={[{...table, normalized_sample: undefined, change_sample: undefined}]}/><DataTable rows={asRows(table.normalized_sample)}/><DataTable rows={asRows(table.change_sample)}/></details>)}
          <DataTable rows={asRows(preview.reject_sample || preview.rejected_rows || preview.errors)} empty="未提供拒收记录"/>
          <p>确认内容绑定本次缓存、映射、结构、目标库和版本头。写入后需完成目标库核验及发布，业务查询才可见。</p>
          <label className="reference-partial"><input type="checkbox" checked={acceptPartial} onChange={event => setAcceptPartial(event.target.checked)}/>允许仅发布通过核验的模块与范围，失败及未知范围继续保留</label>
          {preview.blocking && <div className="error">本次审计存在阻止写入的问题，请修复后重新审计。</div>}
          <button className="primary" disabled={locked || !preview.token || preview.blocking || (rejected > 0 && !acceptPartial) || ['completed', 'rolled_back', 'revoked'].includes(job.status)} onClick={() => run(() => action('commit', {token: preview.token, accept_partial: acceptPartial}))}>确认追加并核验发布</button>
        </div>}
        <h3>分库、分模块写入与发布结果</h3><DataTable rows={asRows(outcome.targets || outcome.results || job.target_status || job.publications || outcome)} testId="reference-publish-results"/>
        <h3>撤销发布记录</h3><DataTable rows={asRows(job.rollbacks)} testId="reference-rollback-results"/>
        <div className="actions"><button disabled={locked} onClick={() => run(() => action('rollback', {targets: requestOf(job).targets || targets}))}>撤销本任务可信发布</button><span className="muted">追加撤销事件，保留全部版本及原审计；受影响的依赖重新检查。</span></div>
      </div>}
    </section>

    <section>
      <h2>参考表结构与版本审计</h2><p>现有生产表和数据保留；结构操作只新增参考版本表和发布元数据。首次使用先预览，再确认建表。</p>
      <DataTable rows={tables} empty="暂无参考表部署信息"/>
      <button disabled={busy || !targets.length} onClick={() => run(async () => {setSchema(await api('/reference/schema/preview', {targets})); setSchemaResult(undefined);})}>预览参考表建表计划</button>
      {schema && <div className="reference-schema"><DataTable rows={asRows(schema.statements || schema.tables || schema.ddl || schema.targets || schema)}/><button className="primary" disabled={busy || !schema.token} onClick={() => run(async () => {setSchemaResult(await api('/reference/schema/apply', {token: schema.token, targets})); setSchema(undefined); const catalog = await api('/reference/tables'); setTables(Array.isArray(catalog) ? catalog : catalog.tables || catalog.rows || []);})}>确认新增参考表结构</button></div>}
      {schemaResult && <DataTable rows={asRows(schemaResult.results || schemaResult.targets || schemaResult)}/>}
      <div className="grid"><label>版本审计目标库<select aria-label="版本审计目标库" value={versionTarget} onChange={event => {setVersionTarget(event.target.value); setVersions(undefined); setVersionSelected('');}}>{Object.entries(targetNames).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label><label>筛选修订编号<input aria-label="筛选修订编号" value={versionSelected} onChange={event => setVersionSelected(event.target.value)}/></label></div>
      <label><input type="checkbox" checked={allVersions} onChange={event => setAllVersions(event.target.checked)}/>审计模式：包含全部版本、未发布记录和撤销历史</label>
      <button disabled={busy} onClick={() => run(async () => setVersions(await api('/reference/versions?target=' + versionTarget + '&audit=' + allVersions)))}>查询参考表版本</button>
      {versions && <div data-testid="reference-versions"><h3>版本与发布范围</h3><DataTable rows={asRows(versions.revisions)}/><DataTable rows={asRows(versions.scopes)}/><h3>版本数据</h3><DataTable rows={selectedVersionRows}/><h3>发布及撤销事件</h3><DataTable rows={asRows(versions.events)}/></div>}
    </section>
  </div>;
}
