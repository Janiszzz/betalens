"""Physical table comparison shared by planning, status, and deployment checks."""
from __future__ import annotations


PG_TYPES = {
    "text": "text",
    "date": "date",
    "timestamp": "timestamp with time zone",
    "integer": "bigint",
}
CH_TYPES = {
    "text": "String",
    "date": "Date32",
    "timestamp": "DateTime64(6,'UTC')",
    "integer": "Int64",
}


def expected_type(field: dict, target: str, key: bool = False) -> str:
    if target == "postgres":
        return (f"numeric({field.get('precision', 38)},{field.get('scale', 12)})"
                if field["type"] == "numeric" else PG_TYPES[field["type"]])
    if target != "clickhouse":
        raise ValueError("目标库无效")
    base = (f"Decimal({field.get('precision', 38)},{field.get('scale', 12)})"
            if field["type"] == "numeric" else CH_TYPES[field["type"]])
    return base if key or not field.get("nullable", True) else f"Nullable({base})"


def _actual_type(field: dict, target: str) -> tuple[str, bool | None]:
    physical = field["physical_type"]
    if target == "postgres":
        if physical == "numeric":
            physical = f"numeric({field.get('precision')},{field.get('scale')})"
        return physical, field.get("nullable")
    normalized = physical.replace(" ", "").replace('"', "'")
    nullable = normalized.startswith("Nullable(") and normalized.endswith(")")
    return (normalized[9:-1] if nullable else normalized), nullable


def compare_structure(spec, actual: dict, target: str) -> dict:
    """Compare type, precision, nullability, field set, and primary/sorting key."""
    fields = spec["fields"] if isinstance(spec, dict) else spec.fields
    keys = list(spec["keys"] if isinstance(spec, dict) else spec.keys)
    current = {field["name"]: field for field in actual.get("fields", [])}
    added, changed, nullable_changes = [], [], []
    for field in fields:
        name = field["name"]
        old = current.get(name)
        if old is None:
            added.append(field)
            continue
        expected = expected_type(field, target, name in keys)
        found_type, found_nullable = _actual_type(old, target)
        expected_base = expected.replace(" ", "")
        if expected_base.startswith("Nullable("):
            expected_base = expected_base[9:-1]
        if found_type.replace(' ','') != expected_base:
            changed.append({"column": name, "from": old["physical_type"], "to": expected})
        required_nullable = name not in keys and field.get("nullable", True)
        if found_nullable is None or found_nullable != required_nullable:
            nullable_changes.append({"column": name, "from": found_nullable,
                                     "to": required_nullable, "physical_type": expected})
    removed = [name for name in current if name not in {field["name"] for field in fields}]
    key_change = bool(actual.get("exists")) and actual.get("keys") != keys
    # PostgreSQL can widen an integer in place while preserving the table OID,
    # grants, indexes and dependencies. It is not a drop/copy/rename migration.
    safe_widenings=[item for item in changed if target=='postgres' and item['from'] in {'smallint','integer'} and item['to']=='bigint']
    requires_rebuild = bool([item for item in changed if item not in safe_widenings] or removed or key_change)
    ready = bool(actual.get("exists")) and not (added or changed or nullable_changes or removed or key_change)
    return {"ready": ready, "added": added, "changed": changed,
            "nullable_changes": nullable_changes, "removed": removed,
            "key_change": key_change, "requires_rebuild": requires_rebuild,"safe_widenings":safe_widenings}
