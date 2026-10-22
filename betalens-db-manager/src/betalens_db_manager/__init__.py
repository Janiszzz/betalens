"""Data infrastructure with no dependency on the Betalens research package."""

__version__ = "0.3.0"

from .config import Settings

__all__ = ["Settings", "__version__"]


from .manager import DatabaseManagerService
from .schema_manager import SchemaManager
__all__ += ["DatabaseManagerService", "SchemaManager"]

from .source_registry import SourceRegistry, SubscriptionAdapter, FetchContext
__all__ += ["SourceRegistry", "SubscriptionAdapter", "FetchContext"]
