import React, {useEffect, useLayoutEffect, useRef, useState} from 'react';

type R = Record<string, any>;
type Props = {
  catalog: R[]; name: string; sources: R[]; document?: R; busy: boolean;
  onTable: (name: string) => void; onChange: (doc: R) => void;
  api: (path: string, body?: any) => Promise<any>;
  run: (fn: () => Promise<any>) => Promise<any>; notice: (message: string) => void;
};
const colors = ['#377dd5', '#13997b', '#9858c5', '#d48724', '#cf5f83', '#318caa'];
const queryId = (b: R) => b.source_id + ':' + b.endpoint;
const emptyTable: R = {keys: [], fields: []};
const producerLabels: Record<string, string> = {direct: '直接映射', constant: '常量', copy: '继承', computed: '计算', system: '系统生成', not_applicable: '不适用', unavailable: '暂无来源'};
const statusLabels: Record<string, string> = {configured: '已配置', unmapped: '真正未映射', not_applicable: '不适用', unavailable: '暂无来源'};
function inputLabel(input: any): string {
  if (typeof input === 'string') return input;
  if (!input || typeof input !== 'object') return String(input ?? '');
  return [input.source_id, input.endpoint || input.table, input.source_field || input.column || input.field].filter(Boolean).join('.');
}
function evidenceItems(column?: R): R[] {return Array.isArray(column?.evidence) ? column!.evidence : [];}
function isVerified(column?: R): boolean {
  const evidence = evidenceItems(column);
  return evidence.length > 0 && evidence.every(e => ['verified_stored', 'verified'].includes(e.status) && Number(e.mismatch_count || 0) === 0);
}
function hasSourceNull(column?: R): boolean {
  return evidenceItems(column).some(e => ['source_null', 'source_all_null'].includes(e.status) || Number(e.source_null_count || e.source_null || 0) > 0);
}

export default function MappingEditor({catalog, name, sources, document: doc, busy, onTable, onChange, api, run, notice}: Props) {
  const table: R = catalog.find(t => t.name === name) || emptyTable;
  const bindings: R[] = doc?.mapping.bindings || [];
  const fields: R[] = (table?.fields || []).filter((f: R) => !f.system && !doc?.protected_columns?.includes(f.name));
  const [active, setActive] = useState(0), [selected, setSelected] = useState('');
  const [expandedSelect, setExpandedSelect] = useState('');
  const [search, setSearch] = useState(''), [sourceSearch, setSourceSearch] = useState('');
  const [pickerOpen, setPickerOpen] = useState(false);
  const picker = useRef<HTMLDivElement>(null);
  const [onlyUnmapped, setOnlyUnmapped] = useState(false), [focus, setFocus] = useState(false);
  const [yaml, setYaml] = useState(''), [issues, setIssues] = useState<R[]>([]), [message, setMessage] = useState('');
  const [coverage, setCoverage] = useState<R>(), [coverageError, setCoverageError] = useState('');
  const [coverageFilter, setCoverageFilter] = useState('all');
  const [evidenceTarget, setEvidenceTarget] = useState('');
  const [showSystem, setShowSystem] = useState(false), [coverageRefresh, setCoverageRefresh] = useState(0);
  const coverageRequest = useRef(0);
  const canvas = useRef<HTMLDivElement>(null);
  const ports = useRef(new Map<string, HTMLSpanElement>());
  const [geometry, setGeometry] = useState<{width: number; height: number; paths: R[]; mapping?: R; table?: string}>({width: 1, height: 1, paths: []});
  const currentSnapshot = useRef({name, mapping: doc?.mapping});
  currentSnapshot.current = {name, mapping: doc?.mapping};
  const visiblePaths = geometry.mapping === doc?.mapping && geometry.table === name ? geometry.paths : [];
  const binding = bindings[active];
  const contract = (b: R) => sources.find(s => s.source_id === b.source_id)?.endpoints?.[b.endpoint];
  const direct = (b: R) => b.mapping_mode === 'direct' || doc?.builtin_table === false;
  const rule = binding?.fields?.find((r: R) => r.source_field === selected);
  const errors = issues.filter(i => i.severity === 'error');
  useEffect(() => {setActive(0); setSelected(''); setSearch(''); setMessage(''); setPickerOpen(false);setCoverageFilter('all');}, [name]);
  useEffect(() => {
    const revision = ++coverageRequest.current;let cancelled = false;
    setCoverage(undefined);setCoverageError('');
    api('/tables/' + encodeURIComponent(name) + '/field-coverage')
      .then(result => {if (!cancelled && revision === coverageRequest.current && result.table === name) setCoverage(result);})
      .catch(error => {if (!cancelled && revision === coverageRequest.current) setCoverageError(error.message || '字段覆盖加载失败');});
    return () => {cancelled = true;};
  }, [name, doc?.file_hash, coverageRefresh]);
  useEffect(() => {
    const outside = (e: PointerEvent) => {if (picker.current && !picker.current.contains(e.target as Node)) setPickerOpen(false);};
    const escape = (e: KeyboardEvent) => {if (e.key === 'Escape') setPickerOpen(false);};
    window.addEventListener('pointerdown', outside);window.addEventListener('keydown', escape);
    return () => {window.removeEventListener('pointerdown', outside);window.removeEventListener('keydown', escape);};
  }, []);
  useEffect(() => {setYaml(doc?.yaml || '');}, [doc?.yaml]);
  useEffect(() => {
    if (!doc) return;
    let cancelled = false;
    const timer = setTimeout(() => api('/tables/' + name + '/mapping/validate', {mapping: doc.mapping})
      .then(r => {if (!cancelled) setIssues(r.diagnostics);})
      .catch(e => {if (!cancelled) setIssues([{severity: 'error', message: e.message}]);}), 250);
    return () => {cancelled = true; clearTimeout(timer);};
  }, [doc?.mapping, name]);

  function change(next: R[]) {
    if (!doc) return;
    const custom = doc?.builtin_table === false;
    onChange({...doc, mapping: {...doc.mapping, format_version: custom ? 2 : doc.mapping.format_version,
      overlap_policy: doc.mapping.overlap_policy || (doc.mapping.bindings.length ? 'priority' : 'reject'), bindings: next.map(b => custom && !b.join_keys ? {...b, join_keys: b.fields.filter((r: R) => table.keys.includes(r.column) && !r.builtin).map((r: R) => ({source_field: r.source_field, column: r.column}))} : b)}});
  }
  function update(index: number, patch: R) {change(bindings.map((b, i) => i === index ? {...b, ...patch} : b));}
  function remove(index: number) {
    change(bindings.filter((_, i) => i !== index));setActive(a => Math.max(0, a > index ? a - 1 : a === index ? 0 : a));setSelected('');
  }
  function toggleSource(s: R, endpoint: string) {
    const index = bindings.findIndex(b => b.source_id === s.source_id && b.endpoint === endpoint);
    if (index >= 0) {remove(index); return;}
    setActive(bindings.length);setSelected('');
    change([...bindings, {source_id: s.source_id, endpoint, mapping_mode: 'direct', priority: bindings.length,
      fields: [], join_keys: [], ignored_fields: {}}]);
  }
  function connect(index: number, source: string, column: string) {
    const b = bindings[index];if (!b || !contract(b)?.fields.some((f: R) => f.name === source)) return;
    if (b.fields.some((r: R) => r.source_field === source && r.builtin)) {setMessage('此字段包含内置计算，请保留计算连线。');return;}
    const occupied = b.fields.find((r: R) => r.column === column && r.source_field !== source);
    if (column && occupied) {setMessage(`${column} 已连接 ${occupied.source_field}，请先移除原连线。`);return;}
    const old = b.fields.find((r: R) => r.source_field === source);
    const next = b.fields.filter((r: R) => r.source_field !== source);
    const ignored = {...b.ignored_fields};delete ignored[source];
    if (column) next.push({...old, source_field: source, column, multiplier: old?.multiplier || '1', required: !!old?.required});
    const joins = next.filter((r: R) => table.keys.includes(r.column) && !r.builtin).map((r: R) => ({source_field: r.source_field, column: r.column}));
    update(index, {fields: next, ignored_fields: ignored, ...(direct(b) ? {join_keys: joins} : {})});
    setActive(index);setSelected(source);setMessage('');
  }
  function ignore(index: number, source: string) {
    const b = bindings[index];const next = b.fields.filter((r: R) => r.source_field !== source);
    update(index, {fields: next, ignored_fields: {...b.ignored_fields, [source]: '用户明确不入此表'},
      ...(direct(b) ? {join_keys: next.filter((r: R) => table.keys.includes(r.column)).map((r: R) => ({source_field: r.source_field, column: r.column}))} : {})});
    setActive(index);setSelected(source);
  }
  function patchRule(patch: R) {update(active, {fields: binding.fields.map((r: R) => r === rule ? {...r, ...patch} : r)});}
  function matches(f: R) {return (f.name + ' ' + (f.zh_name || '') + ' ' + (f.description || '')).toLowerCase().includes(search.toLowerCase());}
  const coverageColumns: R[] = coverage?.table === name && Array.isArray(coverage.columns) ? coverage.columns : [];
  const byColumn = new Map(coverageColumns.map(c => [c.column, c]));
  // Direct links are always read from the current editor. Derived declarations survive all direct-link edits.
  const derived: R[] = (Array.isArray(doc?.mapping.producers) ? doc!.mapping.producers : coverageColumns.flatMap(c => (c.producers || []).map((p: R) => ({...p, column: c.column}))))
    .filter((p: R) => !['direct', 'system', 'not_applicable', 'unavailable'].includes(p.kind));
  const derivedGroups: R[] = [];
  derived.forEach((p, index) => {
    const id = [p.kind, p.processor || '', p.source_id || '', p.endpoint || '', JSON.stringify(p.scope || {})].join(':');
    let group = derivedGroups.find(g => g.id === id);
    if (!group) {group = {id, kind: p.kind, processor: p.processor, endpoint: p.endpoint, rows: []};derivedGroups.push(group);}
    group.rows.push({...p, index});
  });
  function inputPort(input: any): string {
    const label = inputLabel(input);const parts = label.split('.');const field = parts[parts.length - 1];
    const qualified = parts.length > 1 ? parts[parts.length - 2].split(':') : [];
    const endpoint = typeof input === 'object' && input ? input.endpoint : qualified[qualified.length - 1] || '';
    const source = typeof input === 'object' && input ? input.source_id : qualified.length > 1 ? qualified[0] : parts.length > 2 ? parts[0] : '';
    const bi = bindings.findIndex(b => (!source || source === b.source_id) && endpoint === b.endpoint && contract(b)?.fields?.some((f: R) => f.name === field));
    return bi >= 0 ? `s:${bi}:${field}` : 'i:' + label;
  }
  const dependencies = [...new Set(derived.flatMap(p => (p.inputs || []).filter((input: any) => inputPort(input).startsWith('i:')).map(inputLabel)))];
  function producersFor(column: string): R[] {
    const declarations: R[] = Array.isArray(doc?.mapping.producers) ? doc!.mapping.producers : (byColumn.get(column)?.producers || []);
    const links = bindings.flatMap((b, bi) => (b.fields || []).filter((r: R) => r.column === column && !(r.builtin && declarations.some(p => p.column === column && p.kind === 'computed' && p.source_field === r.source_field && p.endpoint === b.endpoint && p.source_id === b.source_id))).map((r: R) => ({...r, kind: r.builtin ? 'computed' : 'direct', endpoint: b.endpoint, source_id: b.source_id, scope: b.scope || {}, bi})));
    return [...links, ...declarations.filter(p => (!p.column || p.column === column) && p.kind !== 'direct')];
  }
  function columnStatus(column: string): string {
    const producers = producersFor(column);
    if (producers.some(p => !['not_applicable', 'unavailable'].includes(p.kind))) return 'configured';
    if (producers.some(p => p.kind === 'not_applicable')) return 'not_applicable';
    if (producers.some(p => p.kind === 'unavailable')) return 'unavailable';
    return byColumn.get(column)?.system || table.fields?.find((field: R) => field.name === column)?.system || doc?.protected_columns?.includes(column) ? 'configured' : 'unmapped';
  }
  function columnEvidence(column: string): R {
    const metadata = byColumn.get(column) || {};
    return {...metadata, evidence: evidenceItems(metadata).filter(e => !evidenceTarget || e.target === evidenceTarget)};
  }
  function matchesCoverage(column: string): boolean {
    const metadata = columnEvidence(column);const status = columnStatus(column);
    if (coverageFilter === 'unmapped') return status === 'unmapped';
    if (coverageFilter === 'not_applicable') return status === 'not_applicable';
    if (coverageFilter === 'unavailable') return status === 'unavailable';
    if (coverageFilter === 'unverified') return !['not_applicable', 'unavailable'].includes(status) && !isVerified(metadata);
    if (coverageFilter === 'source_null') return hasSourceNull(metadata);
    return true;
  }
  function producerDescription(producer: R): string {
    const input = (producer.inputs || []).map(inputLabel).join(' + ');
    if (producer.kind === 'direct') return [producer.endpoint, producer.source_field].filter(Boolean).join('.');
    if (producer.kind === 'constant') return ['值：' + String(producer.value ?? 'NULL'), producer.reason].filter(Boolean).join(' · ');
    return [producer.processor || producer.formula || input, producer.reason].filter(Boolean).join(' · ') || producerLabels[producer.kind] || producer.kind;
  }
  function scopeDescription(scope?: R): string {return Object.entries(scope || {}).map(([key, value]) => key + '=' + (Array.isArray(value) ? value.join('/') : String(value))).join('，');}
  function measure() {
    const root = canvas.current;if (!root || !doc || !table.name) return;
    if (currentSnapshot.current.name !== name || currentSnapshot.current.mapping !== doc?.mapping) return;
    const origin = root.getBoundingClientRect();const paths: R[] = [];
    function point(key: string) {
      const port = ports.current.get(key);if (!port || !port.isConnected) return;
      const r = port.getBoundingClientRect();const clip = port.closest('.diagram-fields')?.getBoundingClientRect();
      if (clip && (r.top < clip.top || r.bottom > clip.bottom)) return;
      return {x: r.left + r.width / 2 - origin.left, y: r.top + r.height / 2 - origin.top};
    }
    bindings.forEach((b, bi) => b.fields.forEach((r: R, ri: number) => {
      if (r.builtin && derived.some(p => p.column === r.column && p.source_field === r.source_field && p.endpoint === b.endpoint && p.source_id === b.source_id)) return;
      const a = point(`s:${bi}:${r.source_field}`), z = point('t:' + r.column);if (!a || !z) return;
      const distance = Math.max(65, (z.x - a.x) / 2);
      paths.push({id: `${bi}:${ri}`, kind: 'direct', bi, endpoint: b.endpoint, source: r.source_field, target: r.column, builtin: r.builtin,
        key: table?.keys.includes(r.column), d: `M ${a.x} ${a.y} C ${a.x + distance} ${a.y}, ${z.x - distance} ${z.y}, ${z.x} ${z.y}`});
    }));
    derived.forEach((producer, index) => {
      function path(from: string, to: string, id: string, source: string, kind: string) {
        const a = point(from), z = point(to);if (!a || !z) return;
        const distance = Math.max(35, Math.abs(z.x - a.x) / 2);
        paths.push({id, kind, source, target: producer.column, builtin: true,
          d: `M ${a.x} ${a.y} C ${a.x + distance} ${a.y}, ${z.x - distance} ${z.y}, ${z.x} ${z.y}`});
      }
      path('p:' + index, 't:' + producer.column, 'producer:' + index, producerLabels[producer.kind] || producer.kind, 'derived');
      (producer.inputs || []).forEach((input: any, i: number) => path(inputPort(input), 'p-in:' + index, `dependency:${index}:${i}`, inputLabel(input), 'dependency'));
    });
    setGeometry({width: root.offsetWidth, height: root.scrollHeight, paths, mapping: doc?.mapping, table: name});
  }
  useLayoutEffect(() => {
    const root = canvas.current;if (!root || !doc || !table.name) return;
    let cancelled = false;
    const frame = requestAnimationFrame(measure);const observer = new ResizeObserver(measure);observer.observe(root);
    window.addEventListener('resize', measure);document.fonts.ready.then(() => {if (!cancelled) measure();});
    return () => {cancelled = true;cancelAnimationFrame(frame);observer.disconnect();window.removeEventListener('resize', measure);};
  }, [doc?.mapping, search, onlyUnmapped, focus, active, table, coverage, coverageFilter]);
  const ref = (key: string) => (node: HTMLSpanElement | null) => {if (node) ports.current.set(key, node);else ports.current.delete(key);};
  const count = bindings.reduce((n, b) => n + b.fields.length, 0);
  const coverageCounts = fields.reduce((result: R, field: R) => {const status = columnStatus(field.name);result[status] = (result[status] || 0) + 1;return result;}, {});

  return <section className="mapping-workbench">
    <div className="heading"><div><h2>本地表与查询源</h2><p className="muted">查询字段 → 共同业务键 → 本地表。连线和共同键一同保存到 YAML。</p></div>
      <button className="primary" disabled={!doc || busy} onClick={() => run(async () => {
        const saved = await api('/tables/' + name + '/mapping', {mapping: doc?.mapping, file_hash: doc?.file_hash});
        onChange(saved);setCoverageRefresh(value => value + 1);notice('映射已写入本地 YAML，新任务使用此版本');
      })}>保存映射规则</button></div>
    <label>选择本地表<select aria-label="选择本地表" value={name} onChange={e => onTable(e.target.value)}>{catalog.map(t => <option key={t.name} value={t.name}>{t.zh_name} · {t.name}</option>)}</select></label>
    {doc?.error && <div className="error">磁盘 YAML 无效，继续使用上一有效版本：{doc.error}</div>}
    <div className="mapping-toolbar">
      <div className="query-picker" ref={picker}><button disabled={!doc || busy} aria-expanded={pickerOpen} aria-haspopup="dialog" onClick={() => setPickerOpen(v => !v)}>选择 API 查询源（已选 {bindings.length}）</button>
        {pickerOpen && <div className="query-menu" role="dialog" aria-label="选择 API 查询源"><input aria-label="搜索 API 查询源" placeholder="搜索订阅源、接口或说明" value={sourceSearch} onChange={e => setSourceSearch(e.target.value)}/>
          <p className="muted">勾选添加，取消勾选移除；同一查询源只登记一次。</p>
          <div role="group" aria-label="API 查询源多选">{sources.map(s => {
            const endpoints = Object.entries(s.endpoints || {}).filter(([endpoint, c]: [string, any]) => (s.label + ' ' + s.source_id + ' ' + endpoint + ' ' + (c.description || '')).toLowerCase().includes(sourceSearch.toLowerCase()));
            return endpoints.length ? <div className="query-group" key={s.source_id}><strong>{s.label} · {s.configured ? '已配置' : '未配置'}</strong>{endpoints.map(([endpoint]) => <label key={endpoint}>
              <input type="checkbox" aria-label={'查询源 ' + s.source_id + ' · ' + endpoint} checked={bindings.some(b => b.source_id === s.source_id && b.endpoint === endpoint)} disabled={!doc || busy || s.source_id === 'wind'} onChange={() => toggleSource(s, endpoint)}/>
              <code>{endpoint}</code><small>{s.endpoints[endpoint].description}</small></label>)}</div> : null;
          })}</div>
        </div>}</div>
      <input aria-label="搜索字段" placeholder="搜索源字段或本地字段" value={search} onChange={e => setSearch(e.target.value)}/>
      <label><input type="checkbox" checked={onlyUnmapped} onChange={e => setOnlyUnmapped(e.target.checked)}/>仅看未处理源字段</label>
      <label><input type="checkbox" checked={focus} onChange={e => setFocus(e.target.checked)}/>仅看当前查询源</label>
    </div>
    <div className="binding-tabs">{bindings.map((b, i) => <div key={queryId(b)} className={active === i ? 'binding-chip active' : 'binding-chip'} style={{borderColor: colors[i % colors.length]}}>
      <button onClick={() => {setActive(i);setSelected('');}}>{b.source_id} · {b.endpoint}</button><button aria-label={'移除查询源 ' + queryId(b)} title="移除此表的查询绑定" onClick={() => remove(i)}>×</button>
    </div>)}</div>
    {!bindings.length && <p className="mapping-empty">从“选择 API 查询源”勾选一个或多个接口，再连接源字段和本地字段。</p>}
    {message && <div role="alert" className="error">{message}</div>}
    <div className="field-coverage-panel">
      <div className="heading"><h3>本地列生产规则与实证</h3><button aria-label="刷新字段覆盖" onClick={() => setCoverageRefresh(value => value + 1)}>刷新覆盖</button></div>
      <p className="muted">“已配置”表示存在生产规则，不代表已有数据验证。计算、继承和系统字段不需要直接源字段连线；空值必须有依据。实证对应已保存规则，编辑后请先保存并重新审计。</p>
      {coverageError ? <p role="alert" className="error">字段覆盖无法加载：{coverageError}。直接映射编辑仍可使用。</p> : !coverage && <p role="status">正在加载已保存规则与字段实证…</p>}
      <div className="coverage-summary" aria-label="本地列覆盖统计">{Object.entries(statusLabels).map(([status, label]) => <span key={status} className={'coverage-tag coverage-' + status}>{label}：{coverageCounts[status] || 0}</span>)}<span className="coverage-tag">未验证：{fields.filter(f => matches(f) && !['not_applicable', 'unavailable'].includes(columnStatus(f.name)) && !isVerified(columnEvidence(f.name))).length}</span></div>
      <div className="coverage-controls"><label>本地字段筛选<select aria-label="本地字段筛选" value={coverageFilter} onChange={event => setCoverageFilter(event.target.value)}><option value="all">全部字段</option><option value="unmapped">真正未映射</option><option value="unverified">未验证</option><option value="source_null">源端空值</option><option value="not_applicable">不适用</option><option value="unavailable">暂无来源</option></select></label>
        <label>实证目标库<select aria-label="实证目标库" value={evidenceTarget} onChange={event => setEvidenceTarget(event.target.value)}><option value="">全部目标库</option><option value="postgres">PostgreSQL</option><option value="clickhouse">ClickHouse</option></select></label>
        <label><input type="checkbox" checked={showSystem} onChange={event => setShowSystem(event.target.checked)}/>覆盖表包含系统列</label>
      </div>
      <details className="column-coverage-details"><summary>逐列生产规则、空值原因与目标库证据</summary><div className="scroll"><table aria-label="本地列字段覆盖"><thead><tr><th>本地列</th><th>生产方式 / 状态</th><th>来源、依赖和留空依据</th><th>适用范围</th><th>目标库</th><th>逐列实证</th></tr></thead><tbody>{(table.fields || []).filter((f: R) => (showSystem || fields.some(column => column.name === f.name)) && matches(f) && matchesCoverage(f.name)).map((field: R) => {
        const producers = producersFor(field.name);const metadata = byColumn.get(field.name);const evidence = evidenceItems(columnEvidence(field.name));const system = metadata?.system || field.system || doc?.protected_columns?.includes(field.name);
        return <tr key={field.name} data-testid={'coverage-' + field.name}><td><strong>{field.name}</strong><small>{field.zh_name} · {field.type} · {field.unit || '无单位'}</small></td><td>{system ? '系统生成' : statusLabels[columnStatus(field.name)]}{producers.map((producer, index) => <small key={index}>{producerLabels[producer.kind] || producer.kind}</small>)}</td><td>{producers.length ? producers.map((producer, index) => <div key={index}>{producerDescription(producer)}{(producer.inputs || []).length > 0 && <small>输入：{producer.inputs.map(inputLabel).join('，')}</small>}</div>) : system ? '由公共入库流程维护，不能映射覆盖' : '尚无有效生产规则；不会按默认零或推断日期补齐'}</td><td>{producers.map((producer, index) => <div key={index}>{scopeDescription(producer.scope) || '通用'}</div>)}</td><td>{evidence.length ? evidence.map((item, index) => <div key={index}>{item.target || '未注明'}</div>) : '无目标库实证'}</td><td>{evidence.length ? evidence.map((item, index) => <div className="column-evidence" key={index}><strong>{item.status || '待核验'}</strong>{item.reason && <small>{item.reason}</small>}{[['适用', 'applicable_count'], ['源空值', 'source_null_count'], ['采用', 'selected_count'], ['写入', 'written_count'], ['保留', 'preserved_count'], ['核验', 'verified_count'], ['不一致', 'mismatch_count']].filter(([, key]) => item[key] !== undefined).map(([label, key]) => <span key={key}>{label} {String(item[key])} </span>)}</div>) : '未验证：没有逐列存储证据；不代表映射错误'}</td></tr>;
      })}</tbody></table></div></details>
    </div>
    <div className="mapping-legend"><span>● 普通字段</span><span>◆ 共同键 / 联合键</span><span>┄ 内置计算 / 继承</span><span>{count} 条字段映射 · {derived.length} 条生产声明 · 当前可见连线 {visiblePaths.length} 条</span><small>滚动节点查看其余字段；被折叠或筛掉的字段仍计入完整性检查。依赖节点只描述输入，不表示向该表写入。</small></div>
    <div className="diagram-viewport" onScrollCapture={measure}>
      <div className={'mapping-canvas' + (derived.length ? ' has-derived-producers' : '')} ref={canvas}>
        <svg className="mapping-edges" width={geometry.width} height={geometry.height} aria-label="字段对应关系连线">
          {visiblePaths.map(p => <g key={p.id} className="mapping-edge" data-testid="mapping-edge" data-source={p.source} data-target={p.target} data-kind={p.kind}>
            <path d={p.d} fill="none" stroke={p.kind === 'direct' ? colors[p.bi % colors.length] : '#8755b4'} strokeWidth={active === p.bi && selected === p.source ? 3 : 1.7} strokeDasharray={p.builtin ? '5 4' : p.key ? '3 2' : undefined} opacity={p.kind !== 'direct' || active === p.bi ? .95 : .42}/>
            <path d={p.d} fill="none" stroke="transparent" strokeWidth={12} style={{pointerEvents: 'stroke', cursor: p.kind === 'direct' ? 'pointer' : 'help'}} onClick={() => {if (p.kind === 'direct') {setActive(p.bi);setSelected(p.source);}}}><title>{p.endpoint ? p.endpoint + '.' : ''}{p.source} → {p.target}{p.key ? '（共同键）' : ''}</title></path>
          </g>)}
        </svg>
        <div className="diagram-sources">{bindings.map((b, bi) => {
          if (focus && active !== bi) return null;
          const c = contract(b);const sourceFields = (c?.fields || []).filter((f: R) => matches(f) && (!onlyUnmapped || (!b.fields.some((r: R) => r.source_field === f.name) && !b.ignored_fields?.[f.name])));
          return <article className={'diagram-node source-node' + (active === bi ? ' node-active' : '')} key={queryId(b)} style={{'--node-color': colors[bi % colors.length]} as React.CSSProperties}>
            <div className="node-heading"><button onClick={() => {setActive(bi);setSelected('');}}><strong>{b.endpoint}</strong><small>{b.source_id} · {c?.fields.length || 0} 个源字段</small></button><span>{direct(b) ? '字段映射' : '内置派生'}</span></div>
            <div className="diagram-fields" onScroll={measure}>{sourceFields.map((f: R) => {
              const rules = b.fields.filter((r: R) => r.source_field === f.name);const ignored = b.ignored_fields?.[f.name];const isKey = rules.some((r: R) => table.keys.includes(r.column));
              return <div key={f.name} className={'diagram-field' + (active === bi && selected === f.name ? ' field-selected' : '') + (ignored ? ' field-ignored' : '')}>
                <div className="field-name"><strong draggable data-testid={'source-' + f.name} onClick={() => {setActive(bi);setSelected(f.name);}}
                  onDragStart={e => {e.dataTransfer.effectAllowed = 'copy';e.dataTransfer.setData('application/x-betalens-mapping', JSON.stringify({binding: bi, source: f.name}));e.dataTransfer.setData('text/plain', f.name);setActive(bi);setSelected(f.name);}}>{isKey ? '◆ ' : '⋮⋮ '}{f.name}</strong><small>{f.type}</small></div>
                <small className="field-description">{f.description || f.name} · {f.source_unit || f.unit || '无单位'}</small>
                {rules.some((r: R) => r.builtin) ? <button className="field-computed" onClick={() => {setActive(bi);setSelected(f.name);}}>查看 {rules.length} 条计算连线</button> : <select aria-label={'映射 ' + f.name} value={ignored ? '__ignore__' : rules[0]?.column || ''} onPointerDown={() => setExpandedSelect(`${bi}:${f.name}`)} onFocus={() => setExpandedSelect(`${bi}:${f.name}`)} onBlur={() => setExpandedSelect('')} onChange={e => e.target.value === '__ignore__' ? ignore(bi, f.name) : connect(bi, f.name, e.target.value)}>
                  <option value="">未映射</option><option value="__ignore__">明确忽略（填写原因）</option>{fields.filter(c => expandedSelect === `${bi}:${f.name}` || rules.some((r: R) => r.column === c.name)).map(f => <option key={f.name} value={f.name} disabled={b.fields.some((r: R) => r.column === f.name && r.source_field !== rules[0]?.source_field)}>{table.keys.includes(f.name) ? '◆ ' : ''}{f.name} · {f.zh_name}</option>)}
                </select>}
                <span className={'field-port source-field-port' + (isKey ? ' key-port' : '')} ref={ref(`s:${bi}:${f.name}`)} onClick={() => {setActive(bi);setSelected(f.name);}} title="拖动字段名或点击此端口，再点击本地字段"/>
              </div>;
            })}</div>
          </article>;
        })}{dependencies.length > 0 && <article className="diagram-node dependency-node" data-testid="mapping-dependency-node"><div className="node-heading"><div><strong>依赖输入</strong><small>由任务计划拉取或已有数据提供，独立于写入目标</small></div><span>只读</span></div><div className="diagram-fields" onScroll={measure}>{dependencies.map(input => <div className="diagram-field dependency-field" key={input}><strong>{input}</strong><span className="field-port source-field-port" ref={ref('i:' + input)}/></div>)}</div></article>}</div>
        {derived.length > 0 && <div className="diagram-producers">{derivedGroups.map(group => <article className="diagram-node producer-node" data-testid="mapping-producer-node" data-kind={group.kind} key={group.id}><div className="node-heading"><div><strong>{producerLabels[group.kind] || group.kind}</strong><small>{group.processor || group.endpoint || '已登记生产规则'}</small></div><span>只读规则</span></div><div className="diagram-fields" onScroll={measure}>{group.rows.filter((producer: R) => matches({name: producer.column, description: producer.reason}) && matchesCoverage(producer.column)).map((producer: R) => <div className="diagram-field producer-field" data-testid={'producer-' + producer.column} key={producer.index}>
          <span className="field-port target-field-port" ref={ref('p-in:' + producer.index)}/><strong>{producer.column}</strong><small>{producerDescription(producer)}</small>{(producer.inputs || []).length > 0 && <small>依赖：{producer.inputs.map(inputLabel).join(' + ')}</small>}{scopeDescription(producer.scope) && <small>适用：{scopeDescription(producer.scope)}</small>}<span className="field-port source-field-port" ref={ref('p:' + producer.index)}/>
        </div>)}</div></article>)}</div>}
        <div className="diagram-target"><article className="diagram-node target-node"><div className="node-heading"><div><strong>{table?.name}</strong><small>{table?.zh_name} · 一行：{table?.grain}</small></div><span>本地表</span></div>
          <div className="diagram-fields" onScroll={measure}>{fields.filter(f => matches(f) && matchesCoverage(f.name)).map(f => {
            const producers = producersFor(f.name);
            return <div key={f.name} className="diagram-field target-field" data-testid={'target-' + f.name} onDragOver={e => e.preventDefault()} onDrop={e => {
              e.preventDefault();let payload = {binding: active, source: e.dataTransfer.getData('text/plain')};
              try {const custom = e.dataTransfer.getData('application/x-betalens-mapping');if (custom) payload = JSON.parse(custom);} catch {return;}connect(payload.binding, payload.source, f.name);
            }} onClick={() => {if (selected && binding) connect(active, selected, f.name);}}>
              <span className={'field-port target-field-port' + (table.keys.includes(f.name) ? ' key-port' : '')} ref={ref('t:' + f.name)}/>
              <div className="field-name"><strong>{table.keys.includes(f.name) ? '◆ ' : ''}{f.name}</strong><small>{f.type}</small></div>
              <small className="field-description">{f.zh_name} · {f.unit || '无单位'} · {f.nullable ? '可空' : '必需'}</small>
              <div className="field-producers">{producers.length ? producers.map((p, i) => <button key={i} className={'producer-' + p.kind} style={{color: p.bi === undefined ? '#8755b4' : colors[p.bi % colors.length]}} title={producerDescription(p) + (scopeDescription(p.scope) ? ' · ' + scopeDescription(p.scope) : '')} onClick={e => {e.stopPropagation();if (p.bi !== undefined) {setActive(p.bi);setSelected(p.source_field);}}}>{producerLabels[p.kind] || p.kind} · {p.kind === 'direct' ? [p.endpoint, p.source_field].filter(Boolean).join('.') : p.processor || p.reason || (p.inputs || []).map(inputLabel).join(' + ') || producerLabels[p.kind]}</button>) : <span className="coverage-unmapped">{table.keys.includes(f.name) ? '共同键待配置' : '真正未映射'}</span>}</div>
              <small className="field-verification">{columnStatus(f.name) === 'not_applicable' ? '该适用范围不产生值' : columnStatus(f.name) === 'unavailable' ? '缺少可靠来源，保留空值' : isVerified(columnEvidence(f.name)) ? '有逐列核验记录' : hasSourceNull(columnEvidence(f.name)) ? '有源端空值记录，详见覆盖表' : '未验证：没有逐列存储证据'}</small>
            </div>;
          })}</div>
        </article></div>
      </div>
    </div>
    {binding && <div className="mapping-properties">
      <h3>当前查询：{queryId(binding)}</h3>
      <div className="grid"><label>资产范围<select aria-label="映射资产范围" value={binding.scope?.asset_type || ''} onChange={e => update(active, {scope: e.target.value ? {asset_type: e.target.value} : {}})}><option value="">通用</option><option value="stock">股票</option><option value="fund">基金</option><option value="index">指数</option></select></label>
        <label>来源优先级（小值优先）<input aria-label="来源优先级" type="number" value={binding.priority || 0} onChange={e => update(active, {priority: Number(e.target.value)})}/></label>
        <label>重复业务列处理<select aria-label="重复业务列处理" value={doc?.mapping.overlap_policy || 'priority'} onChange={e => onChange({...doc, mapping: {...doc?.mapping, overlap_policy: e.target.value}})}><option value="reject">阻止重复，只允许共同键共享</option><option value="priority">显式按优先级主备择取</option></select></label>
      </div>
      <h3>共同键 · {table.keys.length > 1 ? '联合键定位同一行' : '单键定位同一行'}</h3>
      <p className="muted">每个直接来源须完整连接本地业务主键。先转换键的类型和日期格式，再按键值合并；单个接口返回重复键且值不同会阻止入库。</p>
      <div className="join-keys">{table.keys.map((key: string) => {
        const current = binding.fields.find((r: R) => r.column === key && !r.builtin);
        return <label key={key}>◆ {key}{direct(binding) ? <select aria-label={'共同键 ' + key} value={current?.source_field || ''} onChange={e => {
          if (!e.target.value && current) connect(active, current.source_field, '');else if (e.target.value) connect(active, e.target.value, key);
        }}><option value="">请选择源字段</option>{(contract(binding)?.fields || []).map((f: R) => <option key={f.name}>{f.name}</option>)}</select> : <span>{current?.source_field || '由内置处理器生成'} → {key}</span>}</label>;
      })}</div>
      {selected && <div className="connection-properties"><h3>连线设置：{binding.endpoint}.{selected}{rule ? ' → ' + rule.column : ''}</h3>
        {binding.ignored_fields?.[selected] ? <label>忽略原因<input aria-label="忽略原因" value={binding.ignored_fields[selected]} onChange={e => update(active, {ignored_fields: {...binding.ignored_fields, [selected]: e.target.value}})}/></label> : rule ? rule.builtin ? <p>内置计算：{rule.formula}（依赖及计算方式固定）</p> : <>
          <div className="grid"><label>单位乘数<input aria-label={'乘数 ' + selected} value={rule.multiplier || '1'} onChange={e => patchRule({multiplier: e.target.value})}/></label>
            <label>日期格式<input aria-label={'日期格式 ' + selected} value={rule.date_format || ''} placeholder="%Y/%m/%d" onChange={e => patchRule({date_format: e.target.value})}/></label>
            <label>时区<input value={rule.timezone || ''} placeholder="Asia/Shanghai" onChange={e => patchRule({timezone: e.target.value})}/></label></div>
          <label><input type="checkbox" checked={rule.required || false} onChange={e => patchRule({required: e.target.checked})}/>源值必需</label>
          <details><summary>枚举转换</summary>{Object.entries(rule.enum || {}).map(([key, value]) => <div className="enum-row" key={key}>
            <input aria-label={'枚举源值 ' + selected + ' ' + key} value={key} onChange={e => {const next = {...rule.enum};delete next[key];next[e.target.value] = value;patchRule({enum: next});}}/>
            <input aria-label={'枚举本地值 ' + selected + ' ' + key} value={value === null ? '' : String(value)} onChange={e => patchRule({enum: {...rule.enum, [key]: e.target.value || null}})}/>
            <button onClick={() => {const next = {...rule.enum};delete next[key];patchRule({enum: Object.keys(next).length ? next : undefined});}}>移除枚举</button></div>)}
            <button onClick={() => patchRule({enum: {...rule.enum, ['value_' + (Object.keys(rule.enum || {}).length + 1)]: null}})}>增加枚举映射</button></details>
          <button onClick={() => connect(active, selected, '')}>移除当前连线</button>
        </> : <p className="muted">拖动字段名到本地字段，或在源字段下拉框中选择对应列。</p>}
      </div>}
    </div>}
    <div className={'mapping-validation ' + (errors.length ? 'needs-work' : '')}>
      <h3>完整性检查 · {errors.length ? errors.length + ' 项待处理（可保存草稿，不能开始更新）' : bindings.length ? '无阻断项' : '请选择查询源'}</h3>
      {issues.length ? <div className="scroll"><table><thead><tr><th>级别</th><th>查询源</th><th>字段</th><th>检查结果</th></tr></thead><tbody>{issues.map((i, n) => <tr key={n}><td>{i.severity === 'error' ? '待处理' : '提示'}</td><td>{i.source_id} · {i.endpoint}</td><td>{i.field}</td><td>{i.message}</td></tr>)}</tbody></table></div> : <p>共享键允许多条连线；普通列须一一对应，跨源重复列仅在显式配置主备策略后接受。来源发生冲突会进入任务审计。</p>}
    </div>
    <details><summary>本地 YAML 文件与直接编辑</summary><p>{doc?.path}</p><textarea aria-label="映射 YAML" className="code" value={yaml} onChange={e => setYaml(e.target.value)}/><button onClick={() => run(async () => {
      const saved = await api('/tables/' + name + '/mapping', {yaml, file_hash: doc?.file_hash});onChange(saved);setCoverageRefresh(value => value + 1);notice('YAML 已校验并保存');
    })}>校验并保存 YAML</button></details>
  </section>;
}
