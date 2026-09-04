# build_and_entries：逐文件职责与接口

[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)

函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。
TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。

<a id="file-4639989a2b9c"></a>
## .github/workflows/publish.yml

[打开源码](../../../.github/workflows/publish.yml) · 46 行 · 说明来源：人工文件说明

- **作用**：发布 release 后构建并发布 PyPI
- **输入**：GitHub release.published
- **输出**：wheel/sdist 与 PyPI 上传
- **副作用/维护重点**：外部发布；当前工作流没有测试步骤

声明/SQL 操作/配置键定位：

- [L1](../../../.github/workflows/publish.yml#L1)：`name: Publish to PyPI`
- [L3](../../../.github/workflows/publish.yml#L3)：`on:`
- [L7](../../../.github/workflows/publish.yml#L7)：`jobs:`

<a id="file-8b28ba38a2c1"></a>
## .readthedocs.yaml

[打开源码](../../../.readthedocs.yaml) · 25 行 · 说明来源：文件族规则

- **作用**：运行/构建声明式配置
- **输入**：维护者填写的参数
- **输出**：由对应读取器解释的配置对象
- **副作用/维护重点**：文件本身不执行；实际副作用取决于读取它的入口；顶层键见下

声明/SQL 操作/配置键定位：

- [L4](../../../.readthedocs.yaml#L4)：`version: 2`
- [L7](../../../.readthedocs.yaml#L7)：`build:`
- [L13](../../../.readthedocs.yaml#L13)：`sphinx:`
- [L18](../../../.readthedocs.yaml#L18)：`python:`

<a id="file-7ed549aa9f61"></a>
## dashboard/run.bat

[打开源码](../../../dashboard/run.bat) · 19 行 · 说明来源：文件族规则

- **作用**：Windows 启动/初始化包装脚本
- **输入**：当前环境、目录及可用程序
- **输出**：子进程、服务或初始化结果
- **副作用/维护重点**：执行前读脚本：可能启动 GUI、安装依赖或写数据库

<a id="file-0fba722d3629"></a>
## dashboard/run_backend.bat

[打开源码](../../../dashboard/run_backend.bat) · 16 行 · 说明来源：文件族规则

- **作用**：Windows 启动/初始化包装脚本
- **输入**：当前环境、目录及可用程序
- **输出**：子进程、服务或初始化结果
- **副作用/维护重点**：执行前读脚本：可能启动 GUI、安装依赖或写数据库

<a id="file-66247751d19e"></a>
## dashboard/run_frontend.bat

[打开源码](../../../dashboard/run_frontend.bat) · 24 行 · 说明来源：文件族规则

- **作用**：Windows 启动/初始化包装脚本
- **输入**：当前环境、目录及可用程序
- **输出**：子进程、服务或初始化结果
- **副作用/维护重点**：执行前读脚本：可能启动 GUI、安装依赖或写数据库

<a id="file-254a2da7a740"></a>
## docs/conf.py

[打开源码](../../conf.py) · 149 行 · 说明来源：人工文件说明

- **作用**：Sphinx 文档构建配置
- **输入**：Sphinx 与项目 import
- **输出**：HTML 文档构建设置
- **副作用/维护重点**：构建可能触发 autodoc 导入

静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：

```python
from pathlib import Path
import betalens
import os
import sys
```

<a id="file-271b54d579b5"></a>
## docs/requirements.txt

[打开源码](../../requirements.txt) · 7 行 · 说明来源：人工文件说明

- **作用**：文档构建依赖
- **输入**：pip install -r
- **输出**：Sphinx 工具环境
- **副作用/维护重点**：full extras 不等于 docs extras

<a id="file-5d07e7d72637"></a>
## pyproject.toml

[打开源码](../../../pyproject.toml) · 71 行 · 说明来源：人工文件说明

- **作用**：包元数据、依赖与构建配置
- **输入**：pip/build
- **输出**：安装包与 extras
- **副作用/维护重点**：full 不含 test/docs；因子目录不随包发现安装

<a id="file-19359a61ae24"></a>
## requirements.txt

[打开源码](../../../requirements.txt) · 43 行 · 说明来源：人工文件说明

- **作用**：仓库 Python 依赖清单
- **输入**：pip install -r
- **输出**：开发/研究环境依赖
- **副作用/维护重点**：与 pyproject.toml 对照，不能仅凭文件名认定完全锁定版本

