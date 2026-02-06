"""SDK configuration and path resolution."""

import os
from pathlib import Path


class SDKConfig:
    """Configuration for Fraclab SDK paths.

    Resolves SDK home directory from environment variable FRACLAB_SDK_HOME
    or falls back to ~/.fraclab.
    """

    def __init__(self, sdk_home: Path | None = None) -> None:
        """Initialize SDK configuration.

        Args:
            sdk_home: Optional explicit SDK home path. If None, resolves from
                      FRACLAB_SDK_HOME env var or defaults to ~/.fraclab.
        """
        if sdk_home is not None:
            self._sdk_home = Path(sdk_home).expanduser().resolve()
        else:
            env_home = os.environ.get("FRACLAB_SDK_HOME")
            if env_home:
                self._sdk_home = Path(env_home).expanduser().resolve()
            else:
                self._sdk_home = (Path.home() / ".fraclab").expanduser().resolve()

    @property
    def sdk_home(self) -> Path:
        """Root SDK home directory."""
        return self._sdk_home

    @property
    def snapshots_dir(self) -> Path:
        """Directory for snapshot storage."""
        return self._sdk_home / "snapshots"

    @property
    def algorithms_dir(self) -> Path:
        """Directory for algorithm storage."""
        return self._sdk_home / "algorithms"

    @property
    def runs_dir(self) -> Path:
        """Directory for run storage."""
        return self._sdk_home / "runs"

    def ensure_dirs(self) -> None:
        """Create all SDK directories if they don't exist."""
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.algorithms_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
