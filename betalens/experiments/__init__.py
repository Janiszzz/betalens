"""Local experiment archives. Optional database/web dependencies load on demand.

Typical use::

    store = ExperimentStore("outputs/experiments.lmdb")
    factor = store.register_factor(key="momentum", name="动量因子")
    experiment = store.save_run(result, factor_id=factor, config=config)
    analyst = store.to_analyst(experiment)
"""

from .models import ArchiveError, ConflictError, ExperimentInput, MissingArtifactError

__all__ = ["ExperimentStore", "ExperimentInput", "ArchiveError", "ConflictError", "MissingArtifactError"]


def __getattr__(name):
    if name == "ExperimentStore":
        from .store import ExperimentStore
        return ExperimentStore
    raise AttributeError(name)
