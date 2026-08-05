# Betalens Dashboard

Betalens Dashboard 是 Betalens 量化框架的浏览器工作台：FastAPI 后端 + React/Vite 前端。用户可以在浏览器里发现因子、查看 YAML 默认参数、提交回测、看实时日志、浏览指标/图表/持仓/交易明细，并下载 Excel、HTML 和 profiling 产物。

## 启动

前置条件：Python 环境已安装 `.[dashboard,viz]`（或 `.[full]`），Node.js 版本至少为
20.19。先用 `python --version`、`node --version` 和 `npm --version` 验证当前终端环境。

一键启动前后端：

```powershell
.\dashboard\run.bat
```

或分别启动：

```powershell
.\dashboard\run_backend.bat
.\dashboard\run_frontend.bat
```

默认地址：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000`
- Swagger：`http://127.0.0.1:8000/docs`

`run_backend.bat` 会切到仓库根目录，优先使用 `.venv\Scripts\python.exe`，并以 `dashboard.backend.main:app` 启动 Uvicorn。`run_frontend.bat` 会进入 `dashboard/frontend`，缺少 `node_modules` 时自动执行 `npm install`。

## 架构

```text
Browser (React/Vite, :5173)
  | fetch /api/* via Vite proxy
  | EventSource /api/runs/{id}/logs
  v
FastAPI (:8000)
  |- backend/factors.py          扫描 betalens-factor/ 发现因子
  |- backend/runs.py             RunManager + 单线程回测执行队列
  |- backend/serialization.py    backtest/analyst -> JSON + parquet 分页表
  |- backend/eventstudy_dashboard.py
  `- betalens / betalens-factor   真正的取数、分组、回测、评价
```

关键设计：

- 回测任务通过 `ThreadPoolExecutor(max_workers=1)` 串行执行，避免同时抢数据库和内存。
- 每次运行会写出 `outputs/runs/<run_id>/run_config.yaml`，最终运行口径以这份完整 YAML 为准。
- stdout/stderr 被重定向到 `LogBuffer`，前端通过 SSE 实时显示日志。
- 大表不直接塞进结果 JSON；交易和持仓明细落到临时 parquet 后分页读取。
- Excel dump 可能由后台线程异步落盘，下载状态会在结果接口中动态探测。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `backend/main.py` | FastAPI 入口，定义路由和 CORS |
| `backend/factors.py` | 因子发现、详情读取、脚本动态加载 |
| `backend/runs.py` | `RunManager` / `DashboardRun` / `LogBuffer` |
| `backend/serialization.py` | 指标、图表、表格、下载状态序列化 |
| `backend/eventstudy_dashboard.py` | 事件研究文件发现和运行 |
| `backend/schemas.py` | Pydantic 请求/响应模型 |
| `frontend/src/App.tsx` | 单文件 SPA 主界面 |
| `frontend/src/api.ts` | 前端 API 客户端 |
| `frontend/src/types.ts` | 与后端响应对齐的 TS 类型 |
| `frontend/src/styles.css` | 自定义样式 |

## 因子回测数据流

1. 首页请求 `GET /api/factors`。
2. 后端扫描 `betalens-factor/<class>/class_<class>.yaml` 和 `<class>/<name>/factor_<name>.yaml`。
3. 进入详情页后，请求 `GET /api/factors/{class}/{name}`，读取脚本 docstring、默认 run 参数和 `compute_kwargs`。
4. 用户提交运行，前端调用 `POST /api/runs`。
5. 后端合并页面参数和因子 YAML，生成本次 `run_config.yaml`。
6. 后台线程调用 `FactorPipeline(build_spec(...)).run(...)`。
7. 前端轮询 `GET /api/runs/{id}`，并通过 `GET /api/runs/{id}/logs` 接收 SSE 日志。
8. 完成后调用 `GET /api/runs/{id}/result` 获取指标、图表、表格元数据和下载状态。
9. 大表通过 `GET /api/runs/{id}/table/{kind}` 分页读取。

## API 速查

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| GET | `/api/factors?refresh=bool` | 发现因子，`refresh=true` 清缓存 |
| GET | `/api/factors/{class}/{name}` | 因子详情 |
| POST | `/api/runs` | 创建回测运行 |
| DELETE | `/api/runs` | 清空运行缓存 |
| GET | `/api/runs/{id}` | 运行状态 |
| GET | `/api/runs/{id}/logs` | SSE 日志流 |
| GET | `/api/runs/{id}/result` | 指标、图表、表元数据、下载状态 |
| GET | `/api/runs/{id}/table/{kind}` | `trades` / `positions` 分页表 |
| GET | `/api/runs/{id}/download/{kind}` | `dump` / `report` / `html` / `profiling` 下载 |
| GET | `/api/runs/{id}/profiling` | profiling 结果 |
| GET | `/api/eventstudy/files` | 发现事件文件 |
| POST | `/api/eventstudy/run` | 运行事件研究 |

## 新增因子

Dashboard 零改代码发现新因子。目录要求：

```text
betalens-factor/
  <factor_class>/
    class_<factor_class>.yaml
    <FACTOR_NAME>/
      factor_<FACTOR_NAME>.py
      factor_<FACTOR_NAME>.yaml
```

脚本要求：

- import 时不跑回测、不写文件。
- 暴露 `spec`、`FactorPipeline`、`build_spec(config, config_path)`。
- 算子函数参数名匹配 `factor_spec.inputs` 的 key，并接收 `compute_kwargs`。
- CLI 只需要支持 `--config PATH`。

新增或修改 YAML 后，请求 `GET /api/factors?refresh=true` 或重启后端。

## 参数流

页面参数不会多层覆盖运行逻辑。后端会把页面提交的 `parameters` 和 `compute_kwargs` 写入本次 `run_config.yaml`：

```text
RunRequest
  -> RunManager._build_run_config(...)
  -> outputs/runs/<run_id>/run_config.yaml
  -> build_spec(config, config_path)
  -> run_parameters(config, config_path)
  -> FactorPipeline.run(...)
```

因此排查运行口径时优先看 `run_config.yaml`。

## 下载物

- `dump`：`BacktestBase.dump_to_excel()` 产物，包含回测多 sheet 数据。
- `report`：Analyst Excel 报告。
- `html`：plotly 交互 HTML 报告。
- `profiling`：因子体检输出。

## 本地开发

后端：

```powershell
.\dashboard\run_backend.bat
```

前端：

```powershell
.\dashboard\run_frontend.bat
```

前端构建：

```powershell
cd dashboard\frontend
npm run build
```

测试：

```powershell
python -m unittest dashboard.backend.test_factor_yaml dashboard.backend.test_serialization dashboard.backend.test_eventstudy_dashboard
```

## 常见问题

- 首页没有因子：确认 `betalens-factor/` 下存在类级 YAML 和因子级 YAML。
- 修改 YAML 后页面不变：请求 `GET /api/factors?refresh=true` 或重启后端。
- 运行一直排队：后端默认串行执行回测，同一时间只跑一个任务。
- 下载按钮暂时不可用：部分文件异步生成，稍后刷新结果。
- 表格很大但页面不卡：明细表通过 parquet 分页读取，不一次性进入主 JSON。
