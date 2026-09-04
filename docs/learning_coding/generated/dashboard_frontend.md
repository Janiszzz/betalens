# dashboard_frontend：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-14cdcf6290e4"></a>
## dashboard/frontend/package.json

[打开源码](../../../dashboard/frontend/package.json) · 26 行 · 说明来源：人工文件说明

- **作用**：前端依赖与 npm 脚本
- **输入**：npm 命令
- **输出**：dev/build/preview
- **副作用/维护重点**：没有声明 npm test 或 lint 脚本

<a id="file-dc9af101bd32"></a>
## dashboard/frontend/src/App.tsx

[打开源码](../../../dashboard/frontend/src/App.tsx) · 2656 行 · 说明来源：人工文件说明

- **作用**：页面、参数表单、轮询日志与结果展示
- **输入**：API 数据、用户操作、React 状态
- **输出**：页面/图表/任务请求
- **副作用/维护重点**：单文件多组件；修改状态生命周期要清理订阅/轮询

声明/SQL 操作/配置键定位：

- [L43](../../../dashboard/frontend/src/App.tsx#L43)：`const PlotView = lazy(() =&gt; import('./PlotView'));`
- [L45](../../../dashboard/frontend/src/App.tsx#L45)：`type Page = 'home' &#124; 'detail' &#124; 'eventstudy';`
- [L46](../../../dashboard/frontend/src/App.tsx#L46)：`type ResultTab = 'overview' &#124; 'trades' &#124; 'positions' &#124; 'logs';`
- [L48](../../../dashboard/frontend/src/App.tsx#L48)：`const FREQ_LABELS: Record&lt;string, string&gt; = {`
- [L55](../../../dashboard/frontend/src/App.tsx#L55)：`const STATUS_LABELS: Record&lt;string, string&gt; = {`
- [L62](../../../dashboard/frontend/src/App.tsx#L62)：`const WEIGHT_MODE_LABELS: Record&lt;string, string&gt; = {`
- [L67](../../../dashboard/frontend/src/App.tsx#L67)：`const STRATEGY_LABELS: Record&lt;StrategyType, string&gt; = {`
- [L72](../../../dashboard/frontend/src/App.tsx#L72)：`const EXCESS_METRIC_LABELS = new Set([`
- [L94](../../../dashboard/frontend/src/App.tsx#L94)：`const EVENT_FALLBACK_PARAMS: Record&lt;string, unknown&gt; = {`
- [L109](../../../dashboard/frontend/src/App.tsx#L109)：`const formatValue = (metric: Metric) =&gt; {`
- [L117](../../../dashboard/frontend/src/App.tsx#L117)：`const metricGroup = (metric: Metric): 'raw' &#124; 'excess' =&gt; {`
- [L125](../../../dashboard/frontend/src/App.tsx#L125)：`const asString = (value: unknown, fallback = '') =&gt; {`
- [L130](../../../dashboard/frontend/src/App.tsx#L130)：`const asBool = (value: unknown, fallback = false) =&gt; {`
- [L135](../../../dashboard/frontend/src/App.tsx#L135)：`const asNumber = (value: unknown, fallback: number) =&gt; {`
- [L141](../../../dashboard/frontend/src/App.tsx#L141)：`const asNullableNumber = (value: unknown) =&gt; {`
- [L147](../../../dashboard/frontend/src/App.tsx#L147)：`const formatGroupList = (value: unknown) =&gt; {`
- [L153](../../../dashboard/frontend/src/App.tsx#L153)：`const hasGroupList = (value: unknown) =&gt; {`
- [L159](../../../dashboard/frontend/src/App.tsx#L159)：`const isJsonLike = (value: unknown) =&gt; value !== null && typeof value === 'object';`
- [L161](../../../dashboard/frontend/src/App.tsx#L161)：`const prettyJson = (value: unknown) =&gt; JSON.stringify(value ?? {}, null, 2);`
- [L163](../../../dashboard/frontend/src/App.tsx#L163)：`const formatEventDateLabel = (value: unknown) =&gt; {`
- [L170](../../../dashboard/frontend/src/App.tsx#L170)：`const parseEventCodes = (value: unknown) =&gt; Array.from(new Set(`
- [L179](../../../dashboard/frontend/src/App.tsx#L179)：`const EVENT_ASSET_COLORS = [`
- [L190](../../../dashboard/frontend/src/App.tsx#L190)：`type EventCodeSeries = {`
- [L195](../../../dashboard/frontend/src/App.tsx#L195)：`const groupComparisonSeries = (`
- [L222](../../../dashboard/frontend/src/App.tsx#L222)：`const hasOnlyExtremeGroupSelectors = (value: unknown) =&gt; {`
- [L230](../../../dashboard/frontend/src/App.tsx#L230)：`const formatEventAssetLabel = (code: string, assets: EventStudyAsset[] &#124; undefined) =&gt; {`
- [L238](../../../dashboard/frontend/src/App.tsx#L238)：`const formatEventStatsRows = (`
- [L251](../../../dashboard/frontend/src/App.tsx#L251)：`const formatEventStatistic = (value: unknown) =&gt; {`
- [L256](../../../dashboard/frontend/src/App.tsx#L256)：`const formatEventSummaryRows = (`
- [L271](../../../dashboard/frontend/src/App.tsx#L271)：`type PlotRangeBreak = { bounds?: [string, string]; values?: string[] };`
- [L273](../../../dashboard/frontend/src/App.tsx#L273)：`const tradingDayRangebreaks = (records: Array&lt;Record&lt;string, unknown&gt;&gt;): PlotRangeBreak[] =&gt; {`
- [L302](../../../dashboard/frontend/src/App.tsx#L302)：`const buildRunParams = (defaults: Record&lt;string, unknown&gt;) =&gt; ({`
- [L325](../../../dashboard/frontend/src/App.tsx#L325)：`function App() {`
- [L406](../../../dashboard/frontend/src/App.tsx#L406)：`function HomePage({`
- [L467](../../../dashboard/frontend/src/App.tsx#L467)：`function FactorDirectory({`
- [L556](../../../dashboard/frontend/src/App.tsx#L556)：`function FactorPage({`
- [L765](../../../dashboard/frontend/src/App.tsx#L765)：`function StatusBadge({ state }: { state: RunState &#124; null }) {`
- [L771](../../../dashboard/frontend/src/App.tsx#L771)：`function NavButton({ icon, active, onClick, children }: { icon: React.ReactNode; active: boolean; onClick: () =&gt; void; children: React.ReactNode }) {`
- [L775](../../../dashboard/frontend/src/App.tsx#L775)：`function ParameterPanel({`
- [L865](../../../dashboard/frontend/src/App.tsx#L865)：`function JsonInput({`
- [L906](../../../dashboard/frontend/src/App.tsx#L906)：`function LabeledInput({`
- [L936](../../../dashboard/frontend/src/App.tsx#L936)：`function EventStudyPage({ onBack }: { onBack: () =&gt; void }) {`
- [L1096](../../../dashboard/frontend/src/App.tsx#L1096)：`function pickEventFile(`
- [L1110](../../../dashboard/frontend/src/App.tsx#L1110)：`function EventFilePreview({ file }: { file?: EventFile }) {`
- [L1129](../../../dashboard/frontend/src/App.tsx#L1129)：`function EventStudyResultView({ result }: { result: EventStudyResult }) {`
- [L1366](../../../dashboard/frontend/src/App.tsx#L1366)：`function EventStudyComparisonView({`
- [L1510](../../../dashboard/frontend/src/App.tsx#L1510)：`function MetricTile({ label, value, percent = false }: { label: string; value: unknown; percent?: boolean }) {`
- [L1526](../../../dashboard/frontend/src/App.tsx#L1526)：`function eventLayout(title: string, height: number) {`
- [L1537](../../../dashboard/frontend/src/App.tsx#L1537)：`function SimpleTable({ rows, maxHeight = 260 }: { rows: Array&lt;Record&lt;string, unknown&gt;&gt;; maxHeight?: number }) {`
- [L1554](../../../dashboard/frontend/src/App.tsx#L1554)：`function MetricsBlock({ title, metrics }: { title: string; metrics: Metric[] }) {`
- [L1570](../../../dashboard/frontend/src/App.tsx#L1570)：`const timingMetrics = (metrics: Metric[], group: string) =&gt; metrics.filter((metric) =&gt; metric.group === group);`
- [L1572](../../../dashboard/frontend/src/App.tsx#L1572)：`function Overview({ result, state }: { result: RunResult &#124; null; state: RunState &#124; null }) {`
- [L1615](../../../dashboard/frontend/src/App.tsx#L1615)：`function BacktestChartPanels({`
- [L1719](../../../dashboard/frontend/src/App.tsx#L1719)：`function TimingOverview({ result, state }: { result: RunResult &#124; null; state: RunState &#124; null }) {`
- [L1969](../../../dashboard/frontend/src/App.tsx#L1969)：`function formatOptionalPercent(value: unknown) {`
- [L1974](../../../dashboard/frontend/src/App.tsx#L1974)：`function formatOptionalNumber(value: unknown) {`
- [L1980](../../../dashboard/frontend/src/App.tsx#L1980)：`function formatTimingTradeRows(rows: Array&lt;Record&lt;string, unknown&gt;&gt;) {`
- [L1995](../../../dashboard/frontend/src/App.tsx#L1995)：`function formatTimingPredictionRows(rows: Array&lt;Record&lt;string, unknown&gt;&gt;) {`
- [L2004](../../../dashboard/frontend/src/App.tsx#L2004)：`function DiagnosticsPanel({ result }: { result: RunResult }) {`
- [L2027](../../../dashboard/frontend/src/App.tsx#L2027)：`function FactorProfilingPanel({ runId, initial }: { runId: string; initial: FactorProfiling }) {`
- [L2270](../../../dashboard/frontend/src/App.tsx#L2270)：`function baseLayout(title: string, height: number, slider: boolean) {`
- [L2286](../../../dashboard/frontend/src/App.tsx#L2286)：`function RebalanceHoldings({ records }: { records: Array&lt;Record&lt;string, number &#124; string &#124; null&gt;&gt; }) {`
- [L2387](../../../dashboard/frontend/src/App.tsx#L2387)：`function formatPercent(value: number) {`
- [L2392](../../../dashboard/frontend/src/App.tsx#L2392)：`function formatFactor(value: unknown) {`
- [L2401](../../../dashboard/frontend/src/App.tsx#L2401)：`function Downloads({ result }: { result: RunResult }) {`
- [L2416](../../../dashboard/frontend/src/App.tsx#L2416)：`function Trades({ runId, result }: { runId: string &#124; null; result: RunResult &#124; null }) {`
- [L2438](../../../dashboard/frontend/src/App.tsx#L2438)：`function Positions({ runId, result }: { runId: string &#124; null; result: RunResult &#124; null }) {`
- [L2461](../../../dashboard/frontend/src/App.tsx#L2461)：`function ResultTable({`
- [L2567](../../../dashboard/frontend/src/App.tsx#L2567)：`function Pagination({`
- [L2598](../../../dashboard/frontend/src/App.tsx#L2598)：`function formatCell(value: unknown) {`
- [L2604](../../../dashboard/frontend/src/App.tsx#L2604)：`function SearchInput({ value, onChange, placeholder }: { value: string; onChange: (v: string) =&gt; void; placeholder: string }) {`
- [L2608](../../../dashboard/frontend/src/App.tsx#L2608)：`function LabeledInlineInput({`
- [L2627](../../../dashboard/frontend/src/App.tsx#L2627)：`function Logs({ logs, state }: { logs: string; state: RunState &#124; null }) {`
- [L2647](../../../dashboard/frontend/src/App.tsx#L2647)：`function Waiting({ state }: { state: RunState &#124; null }) {`

<a id="file-ff15510d96a5"></a>
## dashboard/frontend/src/PlotView.tsx

[打开源码](../../../dashboard/frontend/src/PlotView.tsx) · 26 行 · 说明来源：人工文件说明

- **作用**：Plotly React 图表包装
- **输入**：data、layout、config props
- **输出**：图表组件
- **副作用/维护重点**：图形渲染和 resize；由 App 懒加载

声明/SQL 操作/配置键定位：

- [L5](../../../dashboard/frontend/src/PlotView.tsx#L5)：`const Plot = createPlotlyComponent(Plotly);`
- [L7](../../../dashboard/frontend/src/PlotView.tsx#L7)：`export default function PlotView({`

<a id="file-4a56ca41d849"></a>
## dashboard/frontend/src/api.ts

[打开源码](../../../dashboard/frontend/src/api.ts) · 72 行 · 说明来源：人工文件说明

- **作用**：HTTP 客户端、参数编码与重试
- **输入**：表单参数、run_id、筛选
- **输出**：Promise<响应类型> 或 Error
- **副作用/维护重点**：网络请求；当前重试包含 POST，注意重复创建任务

声明/SQL 操作/配置键定位：

- [L3](../../../dashboard/frontend/src/api.ts#L3)：`const wait = (ms: number) =&gt; new Promise((resolve) =&gt; window.setTimeout(resolve, ms));`
- [L5](../../../dashboard/frontend/src/api.ts#L5)：`const json = async &lt;T&gt;(url: string, init?: RequestInit, retries = 6): Promise&lt;T&gt; =&gt; {`
- [L36](../../../dashboard/frontend/src/api.ts#L36)：`export const api = {`

<a id="file-df85405b7a37"></a>
## dashboard/frontend/src/main.tsx

[打开源码](../../../dashboard/frontend/src/main.tsx) · 10 行 · 说明来源：人工文件说明

- **作用**：React DOM 挂载
- **输入**：HTML root 元素
- **输出**：App 渲染
- **副作用/维护重点**：浏览器入口

<a id="file-12d876c67cf0"></a>
## dashboard/frontend/src/plotly.d.ts

[打开源码](../../../dashboard/frontend/src/plotly.d.ts) · 10 行 · 说明来源：人工文件说明

- **作用**：Plotly 模块类型声明
- **输入**：TypeScript 编译
- **输出**：类型解析
- **副作用/维护重点**：不是运行时代码

<a id="file-93e49b5a5280"></a>
## dashboard/frontend/src/styles.css

[打开源码](../../../dashboard/frontend/src/styles.css) · 1058 行 · 说明来源：人工文件说明

- **作用**：界面布局和视觉样式
- **输入**：DOM 类名/状态/视口
- **输出**：CSS 样式
- **副作用/维护重点**：类名与 App JSX 对齐

<a id="file-7be4913c0431"></a>
## dashboard/frontend/src/types.ts

[打开源码](../../../dashboard/frontend/src/types.ts) · 184 行 · 说明来源：人工文件说明

- **作用**：前端数据契约
- **输入**：类型声明与后端字段约定
- **输出**：TS 类型约束
- **副作用/维护重点**：编译期类型不能验证运行时 JSON，后端变更需同步

声明/SQL 操作/配置键定位：

- [L1](../../../dashboard/frontend/src/types.ts#L1)：`export type StrategyType = 'cross_sectional' &#124; 'timing';`
- [L3](../../../dashboard/frontend/src/types.ts#L3)：`export type FactorSummary = {`
- [L14](../../../dashboard/frontend/src/types.ts#L14)：`export type FactorDetail = FactorSummary & {`
- [L21](../../../dashboard/frontend/src/types.ts#L21)：`export type RunStatus = 'queued' &#124; 'running' &#124; 'completed' &#124; 'failed';`
- [L23](../../../dashboard/frontend/src/types.ts#L23)：`export type RunState = {`
- [L35](../../../dashboard/frontend/src/types.ts#L35)：`export type Metric = {`
- [L42](../../../dashboard/frontend/src/types.ts#L42)：`export type TimingPayload = {`
- [L60](../../../dashboard/frontend/src/types.ts#L60)：`export type RunResult = {`
- [L93](../../../dashboard/frontend/src/types.ts#L93)：`export type TableMeta = {`
- [L98](../../../dashboard/frontend/src/types.ts#L98)：`export type TablePage = {`
- [L106](../../../dashboard/frontend/src/types.ts#L106)：`export type FactorProfiling = {`
- [L120](../../../dashboard/frontend/src/types.ts#L120)：`export type EventFile = {`
- [L132](../../../dashboard/frontend/src/types.ts#L132)：`export type EventFilesResponse = {`
- [L137](../../../dashboard/frontend/src/types.ts#L137)：`export type EventStudyComparison = {`
- [L157](../../../dashboard/frontend/src/types.ts#L157)：`export type EventStudyAsset = {`
- [L163](../../../dashboard/frontend/src/types.ts#L163)：`export type EventStudyResult = {`

<a id="file-3208b0c1154d"></a>
## dashboard/frontend/tsconfig.json

[打开源码](../../../dashboard/frontend/tsconfig.json) · 21 行 · 说明来源：人工文件说明

- **作用**：前端 TypeScript 编译选项
- **输入**：TS/TSX 源文件
- **输出**：类型检查和编译配置
- **副作用/维护重点**：改严格性或目标会影响 build

<a id="file-cbfe98a289bd"></a>
## dashboard/frontend/tsconfig.node.json

[打开源码](../../../dashboard/frontend/tsconfig.node.json) · 9 行 · 说明来源：人工文件说明

- **作用**：Node/Vite 侧 TypeScript 配置
- **输入**：Vite 等配置源码
- **输出**：Node 侧编译检查选项
- **副作用/维护重点**：与浏览器 tsconfig 职责不同

<a id="file-ea39566ad183"></a>
## dashboard/frontend/vite.config.ts

[打开源码](../../../dashboard/frontend/vite.config.ts) · 70 行 · 说明来源：人工文件说明

- **作用**：Vite、API 代理与后端启动插件
- **输入**：环境变量、Python 路径、开发服务
- **输出**：Vite 配置及开发后端进程
- **副作用/维护重点**：前端启动可能自动启动 Uvicorn；不是单纯静态配置

声明/SQL 操作/配置键定位：

- [L7](../../../dashboard/frontend/vite.config.ts#L7)：`const repoRoot = resolve(__dirname, '../..');`
- [L8](../../../dashboard/frontend/vite.config.ts#L8)：`const backendHost = process.env.DASHBOARD_BACKEND_HOST &#124;&#124; '127.0.0.1';`
- [L9](../../../dashboard/frontend/vite.config.ts#L9)：`const backendPort = process.env.DASHBOARD_BACKEND_PORT &#124;&#124; '8000';`
- [L10](../../../dashboard/frontend/vite.config.ts#L10)：`const backendUrl = process.env.DASHBOARD_BACKEND_URL &#124;&#124; 'http://${backendHost}:${backendPort}';`
- [L11](../../../dashboard/frontend/vite.config.ts#L11)：`const backendEndpoint = new URL(backendUrl);`
- [L12](../../../dashboard/frontend/vite.config.ts#L12)：`const backendSpawnHost = backendEndpoint.hostname &#124;&#124; backendHost;`
- [L13](../../../dashboard/frontend/vite.config.ts#L13)：`const backendSpawnPort = backendEndpoint.port &#124;&#124; backendPort;`
- [L15](../../../dashboard/frontend/vite.config.ts#L15)：`const sleep = (ms: number) =&gt; new Promise((resolveSleep) =&gt; setTimeout(resolveSleep, ms));`
- [L17](../../../dashboard/frontend/vite.config.ts#L17)：`async function waitForBackend(timeoutMs = 30000) {`
- [L31](../../../dashboard/frontend/vite.config.ts#L31)：`function dashboardBackendPlugin() {`

