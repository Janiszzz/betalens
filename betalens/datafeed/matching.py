"""Optional SQL candidate reduction; final matching/formatting stays shared.

Only (code, available_at) winners are reduced, then complete rows at those keys
are fetched. This preserves all same-time conflicts and NULL revisions.
"""
from itertools import product

from .contracts import PROVEN_AVAILABILITY, ACCEPTED_QUALITY, timestamp
from .readers import select_sql, typed_rows


def nearest_candidates(reader, table, fields, filters, codes, anchors, direction, *, ranges=False):
    # Fetching every row at a winning timestamp lets the common business layer
    # detect conflicts. Selecting only a single argMax row would hide them.
    condition = "f.available_at <= i.at" if direction == "before" else "f.available_at > i.at"
    order = "DESC" if direction == "before" else "ASC"
    inputs = []
    for code, anchor in product(dict.fromkeys(codes), anchors):
        start, end = (timestamp(anchor[0]), timestamp(anchor[1])) if ranges else (timestamp(anchor), timestamp(anchor))
        inputs.append((code, start if direction == "after" else end, start, end))
    inputs = list(dict.fromkeys(inputs))
    results = []
    for offset in range(0, len(inputs), 512):
        block = inputs[offset:offset + 512]
        block_filters = list(filters)
        block_filters.extend([("security_code", "IN", list(dict.fromkeys(r[0] for r in block))),
            ("available_at", "IS NOT NULL", None),
            ("availability_basis", "IN", sorted(PROVEN_AVAILABILITY)),
            ("data_quality_status", "IN", sorted(ACCEPTED_QUALITY))])
        source, params = select_sql(reader, table, list(fields), block_filters)
        bound = dict(params)
        if reader.backend == "postgres":
            bound.update(ic=[r[0] for r in block], it=[r[1].to_pydatetime() for r in block],
                         lo=[r[2].to_pydatetime() for r in block], hi=[r[3].to_pydatetime() for r in block])
            input_sql = "SELECT * FROM unnest(%(ic)s::text[],%(it)s::timestamptz[],%(lo)s::timestamptz[],%(hi)s::timestamptz[]) AS x(code,at,lo,hi)"
            interval = ""
            if ranges:
                interval = " AND f.available_at >= i.lo" if direction == "before" else " AND f.available_at < i.hi"
            winners = f"SELECT i.code,hit.available_at FROM ({input_sql}) i CROSS JOIN LATERAL (SELECT f.available_at FROM ({source}) f WHERE f.security_code=i.code AND {condition}{interval} ORDER BY f.available_at {order} LIMIT 1) hit"
        else:
            parts = []
            for n, (code, at, lower, upper) in enumerate(block):
                key = "i" + str(n)
                bound[key] = code
                bound[key + "t"] = at.strftime("%Y-%m-%d %H:%M:%S.%f")
                parts.append("SELECT {" + key + ":String} AS code,{" + key + "t:DateTime64(6,'UTC')} AS at")
            input_sql = " UNION ALL ".join(parts)
            # ASOF only chooses a timestamp; final shared interval filtering
            # removes a nearest hit outside the requested interval.
            winners = f"SELECT i.code, f.available_at FROM ({input_sql}) i ASOF INNER JOIN (SELECT DISTINCT security_code,available_at FROM ({source}) ORDER BY security_code,available_at) f ON i.code=f.security_code AND {condition}"
        quote = '"' if reader.backend == "postgres" else '`'
        projection = ','.join(quote + field + quote for field in fields)
        sql = f"SELECT {projection} FROM ({source}) candidate WHERE (security_code,available_at) IN ({winners})"
        results.extend(typed_rows(reader, table, reader.execute(sql, bound)))
    # Chunk overlap can repeat physical rows; preserve differing row contents.
    unique = {}
    for row in results:
        unique[tuple(row.get(k) for k in fields)] = row
    return list(unique.values())
