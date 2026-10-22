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

export default function MappingEditor({catalog, name, sources, document: doc, busy, onTable, onChange, api, run, notice}: Props) {
  const table: R = catalog.find(t => t.name === name) || emptyTable;
  const bindings: R[] = doc?.mapping.bindings || [];
  const fields: R[] = (table?.fields || []).filter((f: R) => !doc?.protected_columns?.includes(f.name));
  const [active, setActive] = useState(0), [selected, setSelected] = useState('');
  const [expandedSelect, setExpandedSelect] = useState('');
  const [search, setSearch] = useState(''), [sourceSearch, setSourceSearch] = useState('');
  const [pickerOpen, setPickerOpen] = useState(false);
  const picker = useRef<HTMLDivElement>(null);
  const [onlyUnmapped, setOnlyUnmapped] = useState(false), [focus, setFocus] = useState(false);
  const [yaml, setYaml] = useState(''), [issues, setIssues] = useState<R[]>([]), [message, setMessage] = useState('');
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
  useEffect(() => {setActive(0); setSelected(''); setSearch(''); setMessage(''); setPickerOpen(false);}, [name]);
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
      const a = point(`s:${bi}:${r.source_field}`), z = point('t:' + r.column);if (!a || !z) return;
      const distance = Math.max(65, (z.x - a.x) / 2);
      paths.push({id: `${bi}:${ri}`, bi, endpoint: b.endpoint, source: r.source_field, target: r.column, builtin: r.builtin,
        key: table?.keys.includes(r.column), d: `M ${a.x} ${a.y} C ${a.x + distance} ${a.y}, ${z.x - distance} ${z.y}, ${z.x} ${z.y}`});
    }));
    setGeometry({width: root.offsetWidth, height: root.scrollHeight, paths, mapping: doc?.mapping, table: name});
  }
  useLayoutEffect(() => {
    const root = canvas.current;if (!root || !doc || !table.name) return;
    let cancelled = false;
    const frame = requestAnimationFrame(measure);const observer = new ResizeObserver(measure);observer.observe(root);
    window.addEventListener('resize', measure);document.fonts.ready.then(() => {if (!cancelled) measure();});
    return () => {cancelled = true;cancelAnimationFrame(frame);observer.disconnect();window.removeEventListener('resize', measure);};
  }, [doc?.mapping, search, onlyUnmapped, focus, active, table]);
  const ref = (key: string) => (node: HTMLSpanElement | null) => {if (node) ports.current.set(key, node);else ports.current.delete(key);};
  const count = bindings.reduce((n, b) => n + b.fields.length, 0);

  return <section className="mapping-workbench">
    <div className="heading"><div><h2>本地表与查询源</h2><p className="muted">查询字段 → 共同业务键 → 本地表。连线和共同键一同保存到 YAML。</p></div>
      <button className="primary" disabled={!doc || busy} onClick={() => run(async () => {
        const saved = await api('/tables/' + name + '/mapping', {mapping: doc?.mapping, file_hash: doc?.file_hash});
        onChange(saved);notice('映射已写入本地 YAML，新任务使用此版本');
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
      <label><input type="checkbox" checked={onlyUnmapped} onChange={e => setOnlyUnmapped(e.target.checked)}/>仅看未处理字段</label>
      <label><input type="checkbox" checked={focus} onChange={e => setFocus(e.target.checked)}/>仅看当前查询源</label>
    </div>
    <div className="binding-tabs">{bindings.map((b, i) => <div key={queryId(b)} className={active === i ? 'binding-chip active' : 'binding-chip'} style={{borderColor: colors[i % colors.length]}}>
      <button onClick={() => {setActive(i);setSelected('');}}>{b.source_id} · {b.endpoint}</button><button aria-label={'移除查询源 ' + queryId(b)} title="移除此表的查询绑定" onClick={() => remove(i)}>×</button>
    </div>)}</div>
    {!bindings.length && <p className="mapping-empty">从“选择 API 查询源”勾选一个或多个接口，再连接源字段和本地字段。</p>}
    {message && <div role="alert" className="error">{message}</div>}
    <div className="mapping-legend"><span>● 普通字段</span><span>◆ 共同键 / 联合键</span><span>┄ 内置计算</span><span>{count} 条映射 · 当前可见连线 {visiblePaths.length} 条</span><small>滚动节点查看其余字段；被折叠或筛掉的字段仍计入完整性检查。</small></div>
    <div className="diagram-viewport" onScrollCapture={measure}>
      <div className="mapping-canvas" ref={canvas}>
        <svg className="mapping-edges" width={geometry.width} height={geometry.height} aria-label="字段对应关系连线">
          {visiblePaths.map(p => <g key={p.id} className="mapping-edge" data-testid="mapping-edge" data-source={p.source} data-target={p.target}>
            <path d={p.d} fill="none" stroke={colors[p.bi % colors.length]} strokeWidth={active === p.bi && selected === p.source ? 3 : 1.7} strokeDasharray={p.builtin ? '5 4' : p.key ? '3 2' : undefined} opacity={active === p.bi ? .95 : .42}/>
            <path d={p.d} fill="none" stroke="transparent" strokeWidth={12} style={{pointerEvents: 'stroke', cursor: 'pointer'}} onClick={() => {setActive(p.bi);setSelected(p.source);}}><title>{p.endpoint}.{p.source} → {p.target}{p.key ? '（共同键）' : ''}</title></path>
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
        })}</div>
        <div className="diagram-target"><article className="diagram-node target-node"><div className="node-heading"><div><strong>{table?.name}</strong><small>{table?.zh_name} · 一行：{table?.grain}</small></div><span>本地表</span></div>
          <div className="diagram-fields" onScroll={measure}>{fields.filter(f => matches(f) && (!onlyUnmapped || !bindings.some(b => b.fields.some((r: R) => r.column === f.name)))).map(f => {
            const producers = bindings.flatMap((b, i) => b.fields.filter((r: R) => r.column === f.name).map((r: R) => ({...r, bi: i, b})));
            return <div key={f.name} className="diagram-field target-field" data-testid={'target-' + f.name} onDragOver={e => e.preventDefault()} onDrop={e => {
              e.preventDefault();let payload = {binding: active, source: e.dataTransfer.getData('text/plain')};
              try {const custom = e.dataTransfer.getData('application/x-betalens-mapping');if (custom) payload = JSON.parse(custom);} catch {return;}connect(payload.binding, payload.source, f.name);
            }} onClick={() => {if (selected && binding) connect(active, selected, f.name);}}>
              <span className={'field-port target-field-port' + (table.keys.includes(f.name) ? ' key-port' : '')} ref={ref('t:' + f.name)}/>
              <div className="field-name"><strong>{table.keys.includes(f.name) ? '◆ ' : ''}{f.name}</strong><small>{f.type}</small></div>
              <small className="field-description">{f.zh_name} · {f.unit || '无单位'} · {f.nullable ? '可空' : '必需'}</small>
              <div className="field-producers">{producers.length ? producers.map((p, i) => <button key={i} style={{color: colors[p.bi % colors.length]}} onClick={e => {e.stopPropagation();setActive(p.bi);setSelected(p.source_field);}}>{p.b.endpoint}.{p.source_field}</button>) : <span className="muted">{table.keys.includes(f.name) ? '共同键待配置 / 内置派生' : '尚无直接来源'}</span>}</div>
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
      const saved = await api('/tables/' + name + '/mapping', {yaml, file_hash: doc?.file_hash});onChange(saved);notice('YAML 已校验并保存');
    })}>校验并保存 YAML</button></details>
  </section>;
}
