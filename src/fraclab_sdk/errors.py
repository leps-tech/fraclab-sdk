"""SDK error definitions."""

from enum import IntEnum


class ExitCode(IntEnum):
    """CLI exit codes for scripting/CI integration."""

    SUCCESS = 0
    GENERAL_ERROR = 1
    INPUT_ERROR = 2  # Input/parameter errors (validation, path not found)
    RUN_FAILED = 3  # Algorithm execution failed
    TIMEOUT = 4  # Execution timed out
    INTERNAL_ERROR = 5  # Unexpected internal error (bug)


class FraclabError(Exception):
    """Base exception for all Fraclab SDK errors.

    Attributes:
        exit_code: Recommended CLI exit code for this error type.
    """

    exit_code: ExitCode = ExitCode.GENERAL_ERROR


class SnapshotError(FraclabError):
    """Error related to snapshot operations."""

    exit_code = ExitCode.INPUT_ERROR


class AlgorithmError(FraclabError):
    """Error related to algorithm operations."""

    exit_code = ExitCode.INPUT_ERROR


class SelectionError(FraclabError):
    """Error related to selection operations."""

    exit_code = ExitCode.INPUT_ERROR


class MaterializeError(FraclabError):
    """Error related to materialization operations."""

    exit_code = ExitCode.INTERNAL_ERROR


class RunError(FraclabError):
    """Error related to run execution."""

    exit_code = ExitCode.RUN_FAILED


class TimeoutError(RunError):
    """Error when run execution times out."""

    exit_code = ExitCode.TIMEOUT


class ResultError(FraclabError):
    """Error related to result reading."""

    exit_code = ExitCode.INPUT_ERROR


class HashMismatchError(SnapshotError):
    """Error when file hash doesn't match expected hash."""

    def __init__(self, file_name: str, expected: str, actual: str) -> None:
        self.file_name = file_name
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Hash mismatch for {file_name}: expected {expected[:16]}..., got {actual[:16]}..."
        )


class PathTraversalError(FraclabError):
    """Error when a path traversal attempt is detected."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Path traversal detected: {path}")


class DatasetKeyError(SelectionError):
    """Error when a required dataset key is not found."""

    def __init__(self, dataset_key: str, available_keys: list[str]) -> None:
        self.dataset_key = dataset_key
        self.available_keys = available_keys
        super().__init__(
            f"Dataset key '{dataset_key}' not found. Available: {available_keys}"
        )


class CardinalityError(SelectionError):
    """Error when selection violates cardinality constraints."""

    def __init__(
        self, dataset_key: str, cardinality: str, selected_count: int
    ) -> None:
        self.dataset_key = dataset_key
        self.cardinality = cardinality
        self.selected_count = selected_count
        super().__init__(
            f"Cardinality violation for '{dataset_key}': "
            f"cardinality={cardinality}, selected={selected_count}"
        )


class OutputContainmentError(RunError):
    """Error when algorithm attempts to write outside output directory."""

    def __init__(self, attempted_path: str, output_dir: str) -> None:
        self.attempted_path = attempted_path
        self.output_dir = output_dir
        super().__init__(
            f"Output containment violation: attempted to write to {attempted_path}, "
            f"but must be under {output_dir}"
        )
