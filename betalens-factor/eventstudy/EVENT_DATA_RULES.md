# 事件研究数据文件开发标准

本文档描述 `dashboard/backend/eventstudy_dashboard.py` 当前实际扫描规则。新增事件文件必须满足这些约定，才能被事件研究 Dashboard 自动发现和运行。

## 文件位置与格式

事件文件必须放在：

```text
betalens-factor/eventstudy/
```

支持格式：

- `.xlsx`
- `.xls`
- `.csv`

扫描规则：

- 只扫描 `betalens-factor/eventstudy/` 当前目录下的文件。
- 不递归扫描子目录，例如 `eventstudy/data/` 下的文件不会作为事件文件出现。
- 文件按文件名排序展示。
- 文件名会作为前端选择项的 `id`，请避免重名、临时文件名和特殊控制字符。

推荐命名：

```text
1.春节假期.xlsx
2.苹果发布会.xlsx
events.xlsx
```

## 必需字段

事件文件必须包含 `date` 列。

推荐字段：

| 字段 | 必需 | 说明 |
|---|---:|---|
| `date` | 是 | 事件发生时间，可为日期或精确到秒的时间戳 |
| `event` | 否 | 事件标记，`1` 表示事件发生；缺失时后端自动填充为 `1` |
| `remark` | 否 | 事件说明，前端预览会显示 |

最小示例：

| date | event | remark |
|---|---:|---|
| 2020-01-03 | 1 | 美军无人机暗杀伊朗将领苏莱曼尼 |
| 2020-01-08 | 1 | 伊朗导弹报复打击美军驻伊拉克基地 |

## date 字段规则

后端会执行：

```python
pd.to_datetime(df["date"], errors="coerce")
```

规则：

- 支持 `YYYY-MM-DD`。
- 支持 `YYYY-MM-DD HH:MM:SS`。
- 无法解析的日期会被丢弃。
- 事件研究引擎支持精确到秒的事件时点。

时间含义：

- 默认 `market_close_hour=15`。
- 事件发生在 15:00 前：当天收盘价作为 Day 0 成本价。
- 事件发生在 15:00 后：下一交易日收盘价作为 Day 0 成本价。

如果事件只有日期没有时间，Excel 通常会读成 `00:00:00`，即按收盘前事件处理。

## event 字段规则

后端处理逻辑：

```python
if "event" not in df.columns:
    df["event"] = 1
df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int)
events = df[df["event"] == 1]
```

规则：

- `event=1` 的行才会作为事件时点。
- `event=0`、空值、非数值会被视为非事件。
- 如果整份文件没有 `event=1`，运行时会报错。

## remark 字段规则

`remark` 可选，用于前端预览事件说明。

建议写清楚：

- 事件名称
- 事件来源或触发条件
- 事件区间，如节假日、会议期、冲突阶段

示例：

```text
春节假期 2024-02-10 至 2024-02-17
苹果产品发布会
以色列空袭伊朗驻叙利亚大使馆
```

## 其他字段

允许添加其他列，例如：

- `source`
- `category`
- `importance`
- `country`
- `asset`

当前 Dashboard 扫描时会把所有列名放入 `columns`，但预览只显示 `date/event/remark`。其他列不会参与默认分析，除非后续扩展后端。

## Dashboard 运行参数兼容

事件文件只提供事件时点。运行时由前端填写：

- `code`: 标的代码，支持单个代码，也支持用逗号/分号/换行分隔多个代码。
- `benchmark_code`: 基准代码，可选；填写后计算超额收益。
- `metric`: 价格指标，默认 `收盘价(元)`。
- `table_name`: 数据表，默认 `daily_market`。
- `mode`: `flexible` 或 `fixed`。
- `window_before`: 事件前窗口。
- `window_after`: 事件后窗口。
- `holding_start_offset`: 持有起点偏移。
- `market_close_hour`: 收盘小时，默认 15。
- `holding_days`: fixed 模式下的固定持有天数。
- `holding_months`: fixed 模式下的固定持有月数。

注意：

- 事件文件中的日期范围必须落在数据库有行情数据的范围内。
- 标的代码必须在 `table_name` 对应数据表中存在。
- 指标名必须与数据库一致。当前常用值是 `收盘价(元)`，不是 `收盘价`。

## 输出结果

运行成功后，Dashboard 会展示：

- 事件数
- Day 0 平均收益
- Day 0 t 统计
- Day 0 上涨概率
- 事件后窗口末端累积收益
- 累积 t 统计
- 累积上涨概率
- 平均收益柱状图
- 平均累积收益折线图
- 每次事件前后累积收益多折线图
- 三维事件收益矩阵
- 日度统计表
- 累积统计表

底层 `EventStudy.analyze()` 返回：

- `daily_stats`
- `cumulative_stats`
- `returns_matrix`
- `cumulative_returns_matrix`
- `event_count`
- 多标的模式额外返回 `valid_codes` 和 `stock_returns_dict`

## 质量检查清单

提交事件文件前确认：

- 文件位于 `betalens-factor/eventstudy/` 当前目录。
- 文件后缀是 `.xlsx`、`.xls` 或 `.csv`。
- 有 `date` 列。
- 至少有一行 `event=1`，或不提供 `event` 列让后端默认全为事件。
- `date` 能被 pandas 正确解析。
- 事件时点不要重复；如有重复，当前后端不会自动去重。
- 事件日期范围与行情数据库覆盖范围一致。
- `remark` 能让使用者理解事件含义。

## 最小 CSV 示例

```csv
date,event,remark
2020-01-03,1,美军无人机暗杀伊朗将领苏莱曼尼
2020-01-08,1,伊朗导弹报复打击美军驻伊拉克基地
2024-04-01,1,以色列空袭伊朗驻叙利亚大使馆
```
