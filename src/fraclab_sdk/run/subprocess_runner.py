"""Subprocess runner implementation."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"

# Grace period for SIGTERM before escalating to SIGKILL (seconds)
_TERM_GRACE_SECONDS = 2.0


def _kill_process_tree(proc: subprocess.Popen) -> str:
    """Kill a process and its entire tree. Returns the kill strategy used."""
    if _IS_WINDOWS:
        # Windows: try CTRL_BREAK_EVENT first, then kill
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            try:
                proc.wait(timeout=_TERM_GRACE_SECONDS)
                return "ctrl_break"
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass
        proc.kill()
        proc.wait()
        return "kill"
    else:
        # POSIX: use process group kill
        pgid = proc.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=_TERM_GRACE_SECONDS)
                return "killpg_term"
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait()
                return "killpg_kill"
        except OSError:
            # Fallback if process group doesn't exist
            proc.kill()
            proc.wait()
            return "kill"


class SubprocessRunner:
    """Run subprocess with streaming logs and metadata."""

    def __init__(
        self,
        cmd: Iterable[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout_s: int | None = None,
    ) -> None:
        self._cmd = list(cmd)
        self._cwd = Path(cwd)
        self._env = {**os.environ, **(env or {})}
        self._env["PYTHONUNBUFFERED"] = "1"
        self._timeout_s = timeout_s

    def run(self, stdout_path: Path, stderr_path: Path, execute_path: Path) -> tuple[int, bool]:
        """Execute the subprocess, streaming logs and writing metadata.

        Returns:
            (return_code, timed_out)
        """
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        execute_path.parent.mkdir(parents=True, exist_ok=True)

        start_ts = time.time()

        # Platform-specific process group setup
        popen_kwargs: dict = {
            "cwd": self._cwd,
            "env": self._env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "bufsize": 1,
        }
        if _IS_WINDOWS:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(self._cmd, **popen_kwargs)

        timed_out = False
        kill_strategy: str | None = None
        terminated_at: float | None = None

        def _pipe_to_file(pipe, path: Path):
            with path.open("a", encoding="utf-8") as f:
                for line in pipe:
                    f.write(line)
                    f.flush()

        threads: list[threading.Thread] = []
        if proc.stdout:
            t_out = threading.Thread(
                target=_pipe_to_file, args=(proc.stdout, stdout_path), daemon=True
            )
            threads.append(t_out)
            t_out.start()
        if proc.stderr:
            t_err = threading.Thread(
                target=_pipe_to_file, args=(proc.stderr, stderr_path), daemon=True
            )
            threads.append(t_err)
            t_err.start()

        try:
            proc.wait(timeout=self._timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminated_at = time.time()
            kill_strategy = _kill_process_tree(proc)

        for t in threads:
            t.join(timeout=5.0)

        end_ts = time.time()

        meta = {
            "cmd": self._cmd,
            "cwd": str(self._cwd),
            "env": {"PYTHONUNBUFFERED": self._env.get("PYTHONUNBUFFERED", "1")},
            "startedAt": start_ts,
            "endedAt": end_ts,
            "returnCode": proc.returncode,
            "timeout": timed_out,
            "timeoutSeconds": self._timeout_s,
            "killStrategy": kill_strategy,
            "terminatedAt": terminated_at,
        }
        execute_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return proc.returncode or 0, timed_out


__all__ = ["SubprocessRunner"]
