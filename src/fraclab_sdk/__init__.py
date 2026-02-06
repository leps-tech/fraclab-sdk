"""Fraclab SDK - Snapshot, Algorithm, and Run Management."""

from fraclab_sdk.algorithm import AlgorithmHandle, AlgorithmLibrary, AlgorithmMeta
from fraclab_sdk.config import SDKConfig
from fraclab_sdk.materialize import Materializer, MaterializeResult
from fraclab_sdk.results import ResultReader
from fraclab_sdk.run import RunManager, RunMeta, RunResult, RunStatus
from fraclab_sdk.selection.model import SelectionModel
from fraclab_sdk.snapshot import SnapshotHandle, SnapshotLibrary, SnapshotMeta
from fraclab_sdk.version import __version__

__all__ = [
    # Config
    "SDKConfig",
    # Snapshot
    "SnapshotLibrary",
    "SnapshotHandle",
    "SnapshotMeta",
    # Algorithm
    "AlgorithmLibrary",
    "AlgorithmHandle",
    "AlgorithmMeta",
    # Selection
    "SelectionModel",
    # Materialize
    "Materializer",
    "MaterializeResult",
    # Run
    "RunManager",
    "RunMeta",
    "RunResult",
    "RunStatus",
    # Results
    "ResultReader",
    # Package metadata
    "__version__",
]
