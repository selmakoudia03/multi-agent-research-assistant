"""A sandboxed Python execution tool for agent-generated code.

Two independent layers of containment, because LLM-written code is
untrusted input:

1. **Process isolation** — code runs in a fresh `multiprocessing.Process`
   with a hard wall-clock timeout, so an infinite loop cannot hang the
   caller and is killed rather than merely interrupted.
2. **Restricted namespace** — the child process executes with a minimal
   `__builtins__` allowlist (no `__import__`, `open`, `exec`, `eval`,
   `input`, ...), so even within its timeout the code cannot touch the
   filesystem, network, or process table.

This is a defense-in-depth demo sandbox, not a hardened one — it is not a
substitute for OS-level sandboxing (containers, seccomp, gVisor) in a
real production deployment, and that tradeoff is called out in the README.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import multiprocessing
from dataclasses import dataclass

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "max", "min", "print", "range", "round", "set",
        "sorted", "str", "sum", "tuple", "zip",
    )
}


@dataclass
class REPLResult:
    stdout: str
    error: str | None
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out


def _run_in_subprocess(code: str, queue: "multiprocessing.Queue") -> None:
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, "<agent_code>", "exec"), {"__builtins__": _SAFE_BUILTINS}, {})
        queue.put(("ok", buffer.getvalue(), None))
    except Exception as exc:  # noqa: BLE001 - must surface any agent-code error
        queue.put(("error", buffer.getvalue(), f"{type(exc).__name__}: {exc}"))


class PythonREPLTool:
    """Runs a snippet of Python and returns captured stdout, safely."""

    name = "python_repl"

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, code: str) -> REPLResult:
        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue = ctx.Queue()
        process = ctx.Process(target=_run_in_subprocess, args=(code, queue))
        process.start()
        process.join(self.timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            return REPLResult(stdout="", error=None, timed_out=True)

        if queue.empty():
            return REPLResult(stdout="", error="Process exited without output", timed_out=False)

        status, stdout, error = queue.get()
        return REPLResult(stdout=stdout, error=error, timed_out=False)
