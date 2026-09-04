"""静态生成逐文件/逐接口索引；不 import 项目、不连接数据库。

python docs/learning_coding/build_catalog.py
python docs/learning_coding/build_catalog.py --check

只扫描 Git 已跟踪的源码、运行配置和构建脚本。排除本目录、锁文件、
本地配置和产物。--check 只比较，不改文件；修改源码后重新生成索引。
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from file_notes import MIGRATIONS, NOTES

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "generated"
SUFFIXES = {".py", ".ts", ".tsx", ".css", ".sql", ".bat", ".ps1", ".yaml", ".yml", ".toml"}


def selected_files():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    selected = []
    for raw in result.stdout.decode("utf-8").split("\0"):
        path = Path(raw)
        if not raw or raw.startswith("docs/learning_coding/"):
            continue
        if any(part in {"node_modules", "dist", "outputs", "__pycache__", ".venv"} for part in path.parts):
            continue
        if path.name in {"config.json", "config.local.json", "package-lock.json", "uv.lock"}:
            continue
        include = path.suffix in SUFFIXES
        include |= path.name in {"package.json", "config.example.json", "requirements.txt"}
        include |= path.name.startswith("tsconfig") and path.suffix == ".json"
        if include and (ROOT / path).is_file():
            selected.append(path.as_posix())
    return sorted(set(selected))


def group_for(path):
    if path.startswith("betalens/"):
        parts = path.split("/")
        return "core_" + (parts[1] if len(parts) > 2 else "exports")
    if path.startswith("betalens_db_manager/"):
        return "db_migrations" if "/migrations/" in path else "db_manager"
    if path.startswith("dashboard/backend/"):
        return "tests" if Path(path).name.startswith("test_") else "dashboard_backend"
    if path.startswith("dashboard/frontend/"):
        return "dashboard_frontend"
    if re.match(r"betalens-factor/alpha101/ALPHA\d+/", path):
        return "alpha101_catalog"
    if path.startswith("betalens-factor/"):
        return "factor_catalog_and_templates"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("docs/learning/"):
        return "research_course"
    return "build_and_entries"


def describe(path, doc):
    name = Path(path).name
    if path in NOTES:
        parts = [part.strip() for part in NOTES[path].split("|")]
        return dict(zip(("role", "inputs", "outputs", "maintenance"), parts)), "人工文件说明"
    if name in MIGRATIONS and path.endswith(".sql"):
        return {"role": MIGRATIONS[name], "inputs": "迁移前数据库状态；由 SchemaManager 按版本执行",
                "outputs": "DDL/数据迁移后的数据库状态；无 Python 返回值",
                "maintenance": "写库；已有迁移受校验和约束，按源码审查事务与兼容性"}, "人工迁移说明"
    if name == "__init__.py":
        return {"role": doc.split("\n")[0] if doc else "包导出/包标识",
                "inputs": "import 请求", "outputs": "模块导出与符号；见静态 imports",
                "maintenance": "初始化可能导入子模块；__all__ 与真实导出需结合源码阅读"}, "文件族规则"
    if name.startswith("test_"):
        return {"role": "回归测试：" + name, "inputs": "fixture、合成样本与 mock；依赖由测试正文决定",
                "outputs": "断言通过/失败及测试报告", "maintenance": "逐个测试签名和 docstring 见下；测试存在不代表当前已通过"}, "文件族规则"
    if name.startswith("factor_") and path.endswith(".py"):
        return {"role": (doc.split("\n")[0] if doc else name) + "；具体因子脚本/模板",
                "inputs": "完整 YAML、compute 的具名宽表和公式参数；精确签名见下",
                "outputs": "算子结果、spec 或管线 RunResult；运行时可生成报告",
                "maintenance": "先分清算子与 run 入口；同名配置必须完整；不要在 import 时启动回测"}, "文件族规则"
    if path.endswith((".yaml", ".yml")):
        if name.startswith("factor_"):
            role, output = "具体因子的完整运行参数", "meta/factor_spec/weight/run，经配置层转成运行参数"
        elif name.startswith("class_"):
            role, output = "因子类别发现元数据", "类别信息；不能替代完整因子运行 YAML"
        elif name == "parameter_space.yaml":
            role, output = "参数空间、搜索与评价规则", "挖掘候选和窗口配置"
        elif name == "performance.yaml":
            role, output = "挖掘资源、缓存和输出配置", "调度与存储参数"
        else:
            role, output = "运行/构建声明式配置", "由对应读取器解释的配置对象"
        return {"role": role, "inputs": "维护者填写的参数", "outputs": output,
                "maintenance": "文件本身不执行；实际副作用取决于读取它的入口；顶层键见下"}, "文件族规则"
    if path.endswith(".bat"):
        return {"role": "Windows 启动/初始化包装脚本", "inputs": "当前环境、目录及可用程序",
                "outputs": "子进程、服务或初始化结果", "maintenance": "执行前读脚本：可能启动 GUI、安装依赖或写数据库"}, "文件族规则"
    if path.startswith("docs/learning/"):
        return {"role": "研究员课程练习或模板", "inputs": "教学样本/课次参数或研究配置",
                "outputs": "演算输出或模板管线结果", "maintenance": "不是生产入口；真实回测模板与离线实验必须区分"}, "文件族规则"
    return {"role": doc.split("\n")[0] if doc else "源码/配置支持文件，需结合下列声明和调用处阅读",
            "inputs": "见声明参数/配置字段；未在静态索引中推断完整运行时协议",
            "outputs": "见返回注解、源码返回表达式或配置消费入口",
            "maintenance": "无人工专项说明；静态结构不能证明无副作用"}, "源码说明/待专项补充"


class BodySignals(ast.NodeVisitor):
    """只提取当前函数体，不把内部函数的 return 算到外部函数。"""
    def __init__(self):
        self.returns = []
        self.calls = set()

    def visit_FunctionDef(self, node):
        return

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Return(self, node):
        value = ast.unparse(node.value) if node.value is not None else "None"
        self.returns.append({"line": node.lineno, "expression": value})

    def visit_Call(self, node):
        self.calls.add(ast.unparse(node.func))
        self.generic_visit(node)


class Symbols(ast.NodeVisitor):
    def __init__(self):
        self.scope = []
        self.items = []

    def visit_ClassDef(self, node):
        qualified = ".".join([*self.scope, node.name])
        fields = [ast.unparse(item) for item in node.body if isinstance(item, ast.AnnAssign)]
        self.items.append({"name": qualified, "kind": "class", "line": node.lineno,
                           "signature": f"class {node.name}({', '.join(ast.unparse(b) for b in node.bases)})",
                           "docstring": ast.get_docstring(node) or "", "fields": fields,
                           "return_annotation": None, "return_expressions": [], "calls": []})
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        body = BodySignals()
        for statement in node.body:
            body.visit(statement)
        annotation = ast.unparse(node.returns) if node.returns else None
        signature = f"{node.name}({ast.unparse(node.args)})" + (f" -> {annotation}" if annotation else "")
        self.items.append({"name": ".".join([*self.scope, node.name]),
                           "kind": "async function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                           "line": node.lineno, "signature": signature,
                           "docstring": ast.get_docstring(node) or "",
                           "fields": [], "return_annotation": annotation,
                           "return_expressions": body.returns, "calls": sorted(body.calls),
                           "decorators": [ast.unparse(d) for d in node.decorator_list]})
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef


def analyze(path):
    raw = (ROOT / path).read_bytes()
    source = raw.decode("utf-8-sig")
    # 换行规范化，避免 Windows autocrlf 使内容相同的索引过期。
    source = source.replace("\r\n", "\n")
    record = {"path": path, "group": group_for(path), "lines": len(source.splitlines()),
              "sha256": hashlib.sha256(source.encode()).hexdigest(), "symbols": [], "imports": [],
              "declarations": [], "module_docstring": ""}
    if path.endswith(".py"):
        tree = ast.parse(source, filename=path)
        record["module_docstring"] = ast.get_docstring(tree) or ""
        visitor = Symbols()
        visitor.visit(tree)
        record["symbols"] = visitor.items
        record["imports"] = sorted({ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))})
    else:
        for number, line in enumerate(source.splitlines(), 1):
            found = False
            if path.endswith((".ts", ".tsx")):
                found = bool(re.match(r"^(export\s+)?(default\s+)?(async\s+)?(function|class|interface|type|const)\s+", line))
            elif path.endswith(".sql"):
                found = bool(re.match(r"^(CREATE|ALTER|INSERT|UPDATE|DROP|DO|COMMENT)\b", line, re.I))
            elif path.endswith((".yaml", ".yml")):
                found = bool(re.match(r"^[a-zA-Z_][\w.-]*:", line))
            if found:
                record["declarations"].append({"line": number, "text": line.strip()})
    record["notes"], record["description_source"] = describe(path, record["module_docstring"])
    return record


def cell(value, limit=300):
    value = re.sub(r"\s+", " ", str(value)).strip()
    if limit and len(value) > limit:
        value = value[:limit] + "…（完整内容见 inventory.json/源码）"
    return value.replace("|", "&#124;").replace("<", "&lt;").replace(">", "&gt;").replace("`", "'")


def link(path, line=None):
    value = Path(os.path.relpath(ROOT / path, OUTPUT)).as_posix()
    return value + (f"#L{line}" if line else "")


def anchor(path):
    return "file-" + hashlib.sha1(path.encode()).hexdigest()[:12]


def render(records):
    groups = defaultdict(list)
    for item in records:
        groups[item["group"]].append(item)
    fingerprint = hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    payload = {"schema_version": 1, "source_fingerprint": fingerprint,
               "scope": "Git tracked source/config; excludes this guide, locks, secrets and outputs",
               "files": records}
    result = {"inventory.json": json.dumps(payload, ensure_ascii=False, indent=2) + "\n"}
    index = ["# 逐文件源码索引", "", "由 build_catalog.py 静态生成。返回表达式是源码线索，不是推断出的类型或保证。",
             "docstring 保留作者说明，不意味着已验证正确。无注解/无说明的地方请看正文契约与函数体。",
             "", f"覆盖 **{len(records)}** 个文件、**{sum(len(r['symbols']) for r in records)}** 个 Python 类/函数/方法/内部函数声明。",
             "不展开锁文件、研究数据、构建产物；本交接目录本身另在 README 说明。",
             "", "[返回交接导航](../README.md) · [完整机器可读参数、docstring、返回表达式及调用线索](inventory.json)", "",
             "## 分组", "", "| 分组 | 文件数 |", "| --- | ---: |"]
    for group, items in sorted(groups.items()):
        index.append(f"| [{group}]({group}.md) | {len(items)} |")
        lines = [f"# {group}：逐文件职责与接口", "", "[索引](INDEX.md) · [数据形状契约](../02_数据与接口契约.md)", "",
                 "函数表中的‘输出’首先显示注解；无注解时只列 return 表达式。类字段来自源码注解，dataclass/Pydantic 自动构造参数须结合基类阅读。",
                 "TypeScript 声明为正则定位，不是完整 TS 语法解析；不推断运行时输出。", ""]
        for item in items:
            name = item["path"]
            lines += [f'<a id="{anchor(name)}"></a>', f"## {name}", "",
                      f"[打开源码]({link(name)}) · {item['lines']} 行 · 说明来源：{item['description_source']}", ""]
            for title, key in [("作用", "role"), ("输入", "inputs"), ("输出", "outputs"), ("副作用/维护重点", "maintenance")]:
                lines.append(f"- **{title}**：{item['notes'][key]}")
            if item["imports"]:
                lines += ["", "静态 import 线索（包含延迟/条件 import，不等于必然执行依赖）：", "",
                          "```python", *item["imports"], "```"]
            if item["symbols"]:
                lines += ["", "| 符号/位置 | 输入签名或类声明 | 输出注解/返回表达式线索 | 原 docstring 摘要/字段 |",
                          "| --- | --- | --- | --- |"]
                for symbol in item["symbols"]:
                    output = symbol["return_annotation"]
                    if symbol["kind"] == "class":
                        output = "类定义；构造/属性见方法与字段"
                    elif not output:
                        expressions = list(dict.fromkeys(x["expression"] for x in symbol["return_expressions"]))
                        output = "无返回注解；return: " + "; ".join(expressions) if expressions else "无显式 return/返回注解；可能以属性、副作用或 yield 输出"
                    summary = symbol["docstring"] or "无 docstring，需阅读函数体"
                    if symbol["fields"]:
                        summary += "；字段：" + "; ".join(symbol["fields"])
                    lines.append(f"| [{cell(symbol['name'], 0)}]({link(name, symbol['line'])}) | {cell(symbol['signature'], 500)} | {cell(output)} | {cell(summary)} |")
            if item["declarations"]:
                lines += ["", "声明/SQL 操作/配置键定位：", ""]
                for declaration in item["declarations"]:
                    lines.append(f"- [L{declaration['line']}]({link(name, declaration['line'])})：`{cell(declaration['text'])}`")
            lines += [""]
        result[f"{group}.md"] = "\n".join(lines) + "\n"
    index += ["", "## 全文件快速定位", "", "| 文件 | 作用 |", "| --- | --- |"]
    for item in records:
        index.append(f"| [{item['path']}]({item['group']}.md#{anchor(item['path'])}) | {cell(item['notes']['role'])} |")
    result["INDEX.md"] = "\n".join(index) + "\n"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    records = [analyze(path) for path in selected_files()]
    expected = render(records)
    stale = []
    for name, content in expected.items():
        path = OUTPUT / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(name)
        else:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    unexpected = [p.name for p in OUTPUT.glob("*") if p.is_file() and p.name not in expected]
    if stale or unexpected:
        raise SystemExit(f"索引需更新: {stale}; 多余旧产物需人工核对: {unexpected}")
    print(f"{'Checked' if args.check else 'Generated'} {len(records)} files, "
          f"{sum(len(r['symbols']) for r in records)} Python declarations, {len(expected)} catalog artifacts.")
    print("Description sources:", dict(Counter(r["description_source"] for r in records)))


if __name__ == "__main__":
    main()
