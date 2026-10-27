"""Business interval selection is distinct from source announcement filters."""
from .util import parse_date

INTERVALS={'security_industry_membership':('valid_from','valid_to'),
           'industry_membership_record':('in_date','out_date'),
           'security_name_history':('start_date','end_date')}


def row_in_scope(table,row,start,end,time_field=None):
    start,end=parse_date(start),parse_date(end)
    if table in INTERVALS:
        first,last=(row.get(c) for c in INTERVALS[table])
        return (first is None or parse_date(first)<=end) and (last is None or parse_date(last)>=start)
    value=row.get(time_field) if time_field else None
    return value is None or start<=parse_date(value)<=end
