"""Source adapters owned by :mod:`betalens_db_manager`.

These helpers only read or fetch source data and return pandas frames. Database
writes remain the responsibility of :class:`betalens_db_manager.DatabaseWriter`
and :class:`betalens_db_manager.ImportJobRunner`.
"""

from .ede import (
    clean_ede_dataframe,
    extract_date_from_filename,
    extract_date_from_metric_metadata,
    identify_code_name_columns,
    parse_metric_column,
    process_ede_file,
)
from .files import apply_time_alignment, iter_file_chunks, read_csv_with_encoding, read_file
from .industry import build_industry_records
from .wind import (
    fetch_daily_bond,
    fetch_daily_fund,
    fetch_daily_index,
    fetch_daily_market,
)

__all__ = [
    "apply_time_alignment",
    "build_industry_records",
    "clean_ede_dataframe",
    "extract_date_from_filename",
    "extract_date_from_metric_metadata",
    "fetch_daily_bond",
    "fetch_daily_fund",
    "fetch_daily_index",
    "fetch_daily_market",
    "identify_code_name_columns",
    "iter_file_chunks",
    "parse_metric_column",
    "process_ede_file",
    "read_csv_with_encoding",
    "read_file",
]
