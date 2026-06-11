"""Windows compatibility shims for the human_eval package.

human_eval.execution.time_limit relies on signal.setitimer and
signal.SIGALRM, which only exist on Unix. On Windows the per-problem
timeout is enforced by the parent process instead, which joins the
worker with a deadline and kills it on expiry, so the in-process
timer can be replaced by a no-op without losing the timeout
behaviour.

Two further Windows specifics are handled here:

- multiprocessing uses spawn, so a monkeypatch applied in the parent
  does not survive into the worker. The patched check_correctness
  targets _child_unsafe_execute, a wrapper that re-applies the
  time_limit patch inside the child before running the original
  unsafe_execute.
- spawn startup takes seconds, so the parent deadline adds a grace
  period on top of the problem timeout; otherwise every problem is
  misreported as timed out before the worker even starts.
"""

import contextlib
import platform
from collections.abc import Iterator
from multiprocessing.connection import Connection
from typing import Any, Optional

_SPAWN_GRACE_SECONDS = 10.0


@contextlib.contextmanager
def _noop_time_limit(seconds: float) -> Iterator[None]:
    """No-op limit; the parent process enforces the timeout.

    Args:
        seconds: Ignored, kept for signature compatibility.
    """
    yield


def _patch_time_limit() -> None:
    """Replace the Unix-only timer in human_eval.execution."""
    import human_eval.execution as execution

    execution.time_limit = _noop_time_limit


def _child_unsafe_execute(
    problem: dict[str, Any],
    completion: str,
    timeout: float,
    conn: Connection,
) -> None:
    """Worker entry point that patches the child before executing.

    Args:
        problem: HumanEval problem dict.
        completion: Model completion to check.
        timeout: Per-problem timeout in seconds.
        conn: Pipe end used to send the outcome string back.
    """
    _patch_time_limit()
    from human_eval.execution import unsafe_execute

    result: list = []
    unsafe_execute(problem, completion, timeout, result)
    conn.send(result[0] if result else "timed out")
    conn.close()


def apply_windows_patches() -> None:
    """Patch human_eval for Windows. No-op on other platforms."""
    if platform.system() != "Windows":
        return

    from multiprocessing import Pipe, Process

    import human_eval.evaluation as evaluation
    import human_eval.execution as execution

    _patch_time_limit()

    def check_correctness(
        problem: dict[str, Any],
        completion: str,
        timeout: float,
        completion_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """Windows-safe replacement for human_eval check_correctness.

        Uses a Pipe instead of a Manager (one fewer process per call)
        and budgets extra time for spawn startup so the problem
        timeout only measures the test itself.

        Args:
            problem: HumanEval problem dict.
            completion: Model completion to check.
            timeout: Per-problem timeout in seconds.
            completion_id: Optional ID for async result matching.

        Returns:
            Dict with task_id, passed, result, and completion_id.
        """
        recv_conn, send_conn = Pipe(duplex=False)
        p = Process(
            target=_child_unsafe_execute,
            args=(problem, completion, timeout, send_conn),
        )
        p.start()
        send_conn.close()

        outcome = "timed out"
        if recv_conn.poll(timeout + _SPAWN_GRACE_SECONDS):
            try:
                outcome = recv_conn.recv()
            except EOFError:
                outcome = "failed: worker crashed"

        if p.is_alive():
            p.kill()
        p.join()
        recv_conn.close()

        return dict(
            task_id=problem["task_id"],
            passed=outcome == "passed",
            result=outcome,
            completion_id=completion_id,
        )

    execution.check_correctness = check_correctness
    evaluation.check_correctness = check_correctness
