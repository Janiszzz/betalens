# core_exports：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-7ae211babe9c"></a>
## betalens/__init__.py

[打开源码](../../../betalens/__init__.py) · 45 行 · 说明来源：人工文件说明

- **作用**：公共包入口和部分延迟导出
- **输入**：import/属性名
- **输出**：Datafeed、回测及事件研究公共对象
- **副作用/维护重点**：import betalens 成功不证明全部子模块依赖可用

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from .backtest import BacktestBase, BacktestDataError, CodeMismatchError, DateMismatchError
from .datafeed import Datafeed, FillStrategy, get_absolute_trade_days, trade_days_offset
from .eventstudy import EventStudy
```

| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |
| --- | --- | --- | --- |
| [__getattr__](../../../betalens/__init__.py#L27) | __getattr__(name) | 无返回注解；return: values[name]; EventStudy | 无 docstring，需阅读函数体 |

