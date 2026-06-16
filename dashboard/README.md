# betalens Dashboard

新 dashboard 是 FastAPI + React/Vite 的前后端分离应用。

## 启动

后端：

```powershell
.\dashboard\run_backend.bat
```

前端：

```powershell
.\dashboard\run_frontend.bat
```

浏览器打开 `http://127.0.0.1:5173`。

## API

- `GET /api/factors`
- `GET /api/factors/{factor_class}/{name}`
- `POST /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/logs`
- `GET /api/runs/{run_id}/result`
- `GET /api/runs/{run_id}/download/{kind}`
