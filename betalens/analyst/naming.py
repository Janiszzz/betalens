"""
证券代码 → 中文名映射

从数据库各行情表的 name 列取每个 code 最新一条名称，带进程内缓存。
查询失败时静默降级为返回原 code，不阻断分析流程。
"""
import warnings
import pandas as pd

# 候选表（按优先级），覆盖股票/指数/基金/债券
_DEFAULT_TABLES = ["daily_market", "daily_index", "daily_fund", "daily_bond"]

# 进程内缓存：code -> name
_NAME_CACHE: dict = {}


def get_name_map(codes, tables=None) -> dict:
    """
    查库取 code→中文名映射。

    Args:
        codes: 代码可迭代对象
        tables: 候选表列表，None 用默认（股/指/基/债）

    Returns:
        dict: {code: name}，查不到的 code 不在结果中
    """
    codes = [c for c in dict.fromkeys(codes) if c and c != 'cash']
    if not codes:
        return {}

    tables = tables or _DEFAULT_TABLES
    result = {}
    missing = []
    for c in codes:
        if c in _NAME_CACHE:
            if _NAME_CACHE[c] is not None:
                result[c] = _NAME_CACHE[c]
        else:
            missing.append(c)

    if not missing:
        return result

    try:
        from betalens import Datafeed
    except Exception as e:  # pragma: no cover
        warnings.warn(f"无法导入 Datafeed，跳过中文名映射: {e}", UserWarning)
        for c in missing:
            _NAME_CACHE[c] = None
        return result

    found = set()
    for table in tables:
        remaining = [c for c in missing if c not in found]
        if not remaining:
            break
        try:
            db = Datafeed(table)
            try:
                placeholders = ','.join(['%s'] * len(remaining))
                sql = (
                    f"SELECT DISTINCT ON (code) code, name FROM {table} "
                    f"WHERE code IN ({placeholders}) AND name IS NOT NULL "
                    f"ORDER BY code, datetime DESC"
                )
                db.cursor.execute(sql, remaining)
                for row in db.cursor.fetchall():
                    code, name = row['code'], row['name']
                    if name:
                        result[code] = name
                        _NAME_CACHE[code] = name
                        found.add(code)
            finally:
                db.close()
        except Exception:
            # 该表不存在或查询失败，换下一张表
            continue

    # 标记彻底查不到的，避免重复查库
    for c in missing:
        if c not in result:
            _NAME_CACHE.setdefault(c, None)

    return result


def label(code: str, name_map: dict = None) -> str:
    """
    生成展示标签：「中文名(代码)」，无名称时回落为代码本身。

    Example:
        >>> label('000300.SH', {'000300.SH': '沪深300'})
        '沪深300(000300.SH)'
    """
    if code == 'cash':
        return '现金'
    name_map = name_map or {}
    name = name_map.get(code)
    return f"{name}({code})" if name else str(code)


def rename_codes(obj, name_map: dict = None, axis: int = 1):
    """
    把 DataFrame/Series 的 code 索引或列名替换为「中文名(代码)」标签。

    Args:
        obj: DataFrame 或 Series
        name_map: code→name 映射，None 时自动查库
        axis: 1=替换列名（DataFrame），0=替换索引

    Returns:
        替换标签后的副本
    """
    if isinstance(obj, pd.DataFrame):
        codes = obj.columns if axis == 1 else obj.index
    else:
        codes = obj.index
    if name_map is None:
        name_map = get_name_map(list(codes))
    mapping = {c: label(c, name_map) for c in codes}
    if isinstance(obj, pd.DataFrame):
        return obj.rename(columns=mapping) if axis == 1 else obj.rename(index=mapping)
    return obj.rename(index=mapping)


def clear_cache():
    """清空名称缓存（测试或数据更新后调用）"""
    _NAME_CACHE.clear()
