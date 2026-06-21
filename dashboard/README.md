# betalens Dashboard 开发者指南

betalens Dashboard 是 betalens 量化框架的**回测可视化 UI**:FastAPI 后端 + React/Vite 前端,前后端分离。用户在浏览器里发现因子、配参数、跑回测、看净值/指标/持仓、下载报告,真正的计算交给底层 betalens 框架。

---

## 启动

一键起前后端(各开一个窗口):

```powershell
.\dashboard\run.bat
```

或分别启动:

```powershell
.\dashboard\run_backend.bat    # Uvicorn @ 127.0.0.1:8000(--reload)
.\dashboard\run_frontend.bat   # Vite   @ 127.0.0.1:5173(缺 node_modules 自动 npm install)
```

浏览器打开 `http://127.0.0.1:5173`。后端 Swagger 文档:`http://127.0.0.1:8000/docs`。

> [run_backend.bat](run_backend.bat) 会切到**仓库根目录**并优先用 `.venv\Scripts\python.exe`,以 `dashboard.backend.main:app` 启动。

---

## 架构总览

```
浏览器 (App.tsx, :5173)
   │  fetch /api/*  ──(Vite dev 代理)──▶  FastAPI (:8000)
   │  EventSource /api/runs/{id}/logs (SSE)
   ▼
FastAPI [backend/main.py]  ── CORS 仅放行 5173
   ├─ factors.py      扫描 betalens-factor/ 发现因子
   ├─ runs.py         RunManager(单例)+ ThreadPoolExecutor(max_workers=1)
   │                     └─ DashboardRun ── _execute()
   │                            └─▶ factor_template.FactorPipeline(spec).run(...)
   │                                   └─▶ betalens/  (datafeed→factor→backtest→analyst)
   └─ serialization.py  把 backtest/analyst 转成 JSON(metrics/charts/table meta)
                         并把 trades/positions 落到临时 parquet 后分页读取
```

关键设计:回测在**单线程池**里串行执行(同一时刻只跑一个),不阻塞 API;执行期间 stdout/stderr 被重定向并通过 **SSE** 实时推给前端;完成后主结果只保留指标、图表和表元数据,交易/持仓明细走分页接口;Excel dump 由独立后台线程异步落盘。

---

## 目录结构

| 路径 | 说明 |
|------|------|
| [backend/main.py](backend/main.py) | FastAPI 入口,定义全部路由 + CORS |
| [backend/factors.py](backend/factors.py) | 因子发现/详情/动态加载(`importlib`) |
| [backend/runs.py](backend/runs.py) | `RunManager` / `DashboardRun` / `LogBuffer`,回测执行与日志 |
| [backend/serialization.py](backend/serialization.py) | 回测结果 → JSON(指标、图表、表元数据),表明细 parquet 分页 |
| [backend/schemas.py](backend/schemas.py) | Pydantic 请求/响应模型 |
| [frontend/src/App.tsx](frontend/src/App.tsx) | 单文件 SPA:Home(因子列表)+ Detail(参数/结果) |
| [frontend/src/api.ts](frontend/src/api.ts) | Fetch 封装的后端客户端 |
| [frontend/src/types.ts](frontend/src/types.ts) | 与后端对齐的 TS 类型 |
| [frontend/src/styles.css](frontend/src/styles.css) | 自定义设计系统(无第三方 UI 库) |
| `logs/` | 本地运行时生成的 datafeed 日志,不入库 |

技术栈:后端 FastAPI + Uvicorn + Pydantic + PyArrow;前端 React 19 + Vite 7 + TS,**无路由库/状态库/UI 库**,用 `useState` 管页面、Plotly 画图、`lucide-react` 图标。

---

## 端到端数据流

以「跑一次因子回测」为例,四步走:

**① 因子发现** — 进首页加载列表
```
api.factors()  →  GET /api/factors  →  discover_factors()
```
`discover_factors()`([factors.py:43](backend/factors.py#L43))扫描 `betalens-factor/<class>/spec_<class>.json`,展开每个因子为 `FactorSummary`,用 `@lru_cache` 缓存。

**② 看因子详情** — 点卡片进详情页
```
api.factor(cls, name)  →  GET /api/factors/{class}/{name}  →  get_factor_detail()
```
`get_factor_detail()`([factors.py:78](backend/factors.py#L78))定位 `<class>/<name>/factor_<name>.py`,`importlib` 加载取 docstring 与 `compute_kwargs`。

**③ 提交回测** — 配好参数点运行
```
api.startRun(body)  →  POST /api/runs  →  RunManager.create()  →  线程池 _execute()
```
`_execute()`([runs.py:140](backend/runs.py#L140))合并参数进 `factor_spec`(`dataclasses.replace`),把 stdout 重定向到 `LogBuffer`,调用:
```python
FactorPipeline(factor_spec).run(start, end,
    rebal_freq=..., n_quantiles=..., initial_amount=...,
    output_dir=..., include_profiling=...)
```
内部依次驱动 betalens 的 datafeed(取数)→ factor(分组/中性化)→ backtest(交易模拟)→ analyst(绩效)。

**④ 看进度与结果** — 前端边轮询边收日志
```
轮询: setInterval 1200ms → api.run(id) → GET /api/runs/{id}          # status
日志: EventSource → GET /api/runs/{id}/logs                          # SSE 增量
完成: api.result(id) → GET /api/runs/{id}/result → serialize_result  # 指标+图表+表元数据
翻页: api.table(id, kind, opts) → GET /api/runs/{id}/table/{kind}     # trades/positions 分页
下载: GET /api/runs/{id}/download/{kind}                              # xlsx/html
```

---

## API 速查

对照 [backend/main.py](backend/main.py):

| 方法 | 路径 | 入参 | 返回 | 错误 |
|------|------|------|------|------|
| GET | `/api/health` | — | `{status:"ok"}` | — |
| GET | `/api/factors` | `?refresh=bool` | `FactorSummary[]` | — |
| GET | `/api/factors/{class}/{name}` | path | `FactorDetail` | 404 不存在 |
| POST | `/api/runs` | `RunRequest` | `{run_id}` | 400 创建失败 |
| GET | `/api/runs/{id}` | path | `RunState` | 404 |
| GET | `/api/runs/{id}/logs` | path | SSE 流 `event: log/close` | 404 |
| GET | `/api/runs/{id}/result` | path | 指标、图表、表元数据、下载状态 | 404 / 409 未完成 |
| GET | `/api/runs/{id}/table/{kind}` | `kind∈{trades,positions}`, `page`, `size`, `query`, `filter.<col>` | 分页表格 | 404 / 409 |
| GET | `/api/runs/{id}/download/{kind}` | `kind∈{dump,report,html,profiling}` | 文件 | 404 |

### 后端函数速查

| 函数 | 位置 | 职责与易错点 |
|------|------|------|
| `discover_factors()` | [factors.py:43](backend/factors.py#L43) | 扫描并缓存因子;**改了 spec 要 `?refresh=true` 或重启**才生效 |
| `get_factor_config()` | [factors.py:63](backend/factors.py#L63) | 校验 spec/脚本存在,找不到抛 `FileNotFoundError`→404 |
| `load_factor_module()` | [factors.py:101](backend/factors.py#L101) | `importlib` 动态加载;会把 `REPO_ROOT/factor_root/class_dir` 插入 `sys.path` |
| `RunManager.create()` | [runs.py:126](backend/runs.py#L126) | 建 `DashboardRun` 提交线程池;**`max_workers=1` 串行** |
| `RunManager._execute()` | [runs.py:140](backend/runs.py#L140) | 合并参数→跑 `FactorPipeline`→表落 parquet→释放 backtest/analyst→异步 dump;异常进 `mark_failed` |
| `serialize_result()` | [runs.py:242](backend/runs.py#L242) | run 必须完成;返回缓存 payload 并实时探测 downloads |
| `read_table_page()` | [serialization.py](backend/serialization.py) | 从 parquet 读取 trades/positions,支持分页、全文搜索和列过滤 |

### 前端类型速查

见 [frontend/src/types.ts](frontend/src/types.ts):`FactorSummary`(列表项)、`FactorDetail`(+doc/script_path)、`RunState`(status/elapsed/error)、`RunResult`(run/factor/metrics/charts/tables/downloads)、`TablePage`(分页行)。前后端字段必须一一对齐。

---

## 二次开发指引

**加一个因子(零改 dashboard 代码)**
在 [betalens-factor/](../betalens-factor/) 下:
```
<class>/spec_<class>.json          # 声明 class/source/defaults/factors[]
<class>/<name>/factor_<name>.py    # 暴露模块级 spec 对象
```
脚本须能被 `importlib` 独立加载(顶层别依赖未在 `sys.path` 的相对导入)。新增后调 `GET /api/factors?refresh=true` 或重启后端清 `lru_cache`。

**加一个回测参数**
```
ParameterPanel 加输入(App.tsx)
  → RunRequest.parameters 透传(schemas.py)
  → runs._execute 接住:落在 spec_updates 白名单 或 FactorPipeline.run 的 kwargs
```
白名单当前为 `direction/index_code/use_industry/use_mktcap/industry_scheme/backtest_metric/compute_kwargs`([runs.py:149](backend/runs.py#L149));`rebal_freq/n_quantiles/initial_amount/include_profiling` 走 `kwargs`([runs.py:180](backend/runs.py#L180))。

**加一个 API 端点**
```
main.py 加路由  +  schemas.py 加 Pydantic 模型
  +  frontend/src/api.ts 加客户端方法  +  types.ts 加类型
```
注意:浏览器直连的新端点要确保在 CORS 白名单([main.py:21](backend/main.py#L21))内。

**加一个结果图表/指标**
```
serialization.build_chart_data / build_metrics 加字段
  → types.ts 的 RunResult 加字段
  → App.tsx 的 Overview 加 Plotly 图或指标卡
```

---

## 本地开发与调试

- **改后端**:`--reload` 自动重启;`http://127.0.0.1:8000/docs` 直接试接口。
- **改前端**:Vite HMR 即时刷新;`/api` 已代理到 8000,无需配 baseURL。
- **构建前端**:`cd frontend && npm run build`(先 `tsc -b` 类型检查,再出 `dist/`)。
- **后端测试**:在仓库根目录运行 `python -m unittest dashboard.backend.test_serialization`。
- **日志和缓存**:回测日志走 SSE 实时显示在详情页;本地 `logs/` 是 datafeed 查询日志;表格 parquet 缓存在系统临时目录 `betalens_dashboard_runs/`,LRU 淘汰 run 时自动清理。
