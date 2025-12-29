"""Safe code execution utilities for untrusted Python code.

This module provides sandboxed execution of model-generated code with safety measures:
- Timeout protection via signal-based interrupts
- IO isolation to prevent information leakage
- Resource limits to prevent system abuse
- Privilege reduction for process isolation

Adapted from OpenAI's human-eval execution with additional safety hardening.

SECURITY NOTICE: This is NOT cryptographic security. Malicious code may still
cause damage. Use in isolated environments with proper system-level protections.
"""

from __future__ import annotations

import atexit
import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import sys
import tempfile
import warnings
from typing import Any


# Suppress noisy warnings from multiprocessing child process cleanup
# These are harmless but clutter logs significantly
def _suppress_execution_sandbox_noise() -> None:
    """Configure filters to suppress expected sandbox cleanup noise.

    The execution sandbox spawns child processes that produce harmless but
    noisy error messages during cleanup:
    - EOFError in multiprocessing logging queue monitor threads
    - TemporaryDirectory cleanup TypeError when os.unlink is disabled
    - asyncio "Task was destroyed" warnings for websocket connections

    These are expected behavior and don't indicate real problems.
    """
    # Suppress ResourceWarnings from unclosed resources during sandbox exit
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Suppress DeprecationWarnings from sandbox child processes
    warnings.filterwarnings("ignore", category=DeprecationWarning)


class TimeoutException(Exception):
    """Raised when code execution exceeds timeout limit."""

    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO wrapper that prevents reading to avoid information leakage."""

    def read(self, *args: Any, **kwargs: Any) -> str:
        """Disabled read operation."""
        raise io.UnsupportedOperation("not readable")

    def readline(self, *args: Any, **kwargs: Any) -> str:
        """Disabled readline operation."""
        raise io.UnsupportedOperation("not readable")

    def readlines(self, *args: Any, **kwargs: Any) -> list[str]:
        """Disabled readlines operation."""
        raise io.UnsupportedOperation("not readable")

    def readable(self) -> bool:
        """Mark stream as write-only."""
        return False


@contextlib.contextmanager
def time_limit(seconds: float):
    """Context manager to enforce execution timeout using SIGALRM.

    Args:
        seconds: Maximum execution time in seconds

    Raises:
        TimeoutException: If execution exceeds time limit

    Note:
        Only works on Unix-like systems. Windows timeout handled at process level.
    """

    def signal_handler(signum: int, frame: Any) -> None:
        raise TimeoutException(f"Code execution timed out after {seconds} seconds")

    if platform.system() != "Windows":
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(seconds))
    try:
        yield
    finally:
        if platform.system() != "Windows":
            signal.alarm(0)


@contextlib.contextmanager
def swallow_io():
    """Context manager to suppress all IO operations.

    Redirects stdout/stderr to write-only buffers and disables stdin.
    Prevents code from reading system state or leaking information.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_stdin = sys.stdin

    sys.stdout = WriteOnlyStringIO()
    sys.stderr = WriteOnlyStringIO()
    sys.stdin = None  # type: ignore[assignment]

    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin = old_stdin


@contextlib.contextmanager
def create_tempdir():
    """Context manager for temporary directory creation and cleanup."""
    with tempfile.TemporaryDirectory() as dirname:
        old_cwd = os.getcwd()
        try:
            os.chdir(dirname)
            yield dirname
        finally:
            os.chdir(old_cwd)


def reliability_guard(max_memory_bytes: int = 2 * 1024**3) -> None:
    """Configure resource limits and disable dangerous operations.

    Args:
        max_memory_bytes: Maximum memory in bytes (default: 2GB)

    Disables:
        - Dangerous system calls (os.kill, os.system, etc.)
        - Subprocess creation
        - Problematic module imports (ipdb, joblib, etc.)
        - File system modifications (shutil.rmtree, os.remove)

    Sets:
        - Memory limits (address space)
        - Prevents core dumps
    """
    # Apply noise suppression for sandbox child processes
    _suppress_execution_sandbox_noise()

    # Disable faulthandler to prevent info leakage via core dumps
    faulthandler.disable()

    # Set resource limits on Unix systems
    if platform.system() != "Windows":
        import resource

        # Limit memory usage (address space)
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))

        # Disable core dumps
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        # Limit number of file descriptors
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    # Disable dangerous built-in functions
    import builtins

    builtins.exit = None  # type: ignore[attr-defined]
    builtins.quit = None  # type: ignore[attr-defined]

    # Save originals for internal cleanup before disabling
    _original_unlink = os.unlink
    _original_rmdir = os.rmdir

    # Register atexit handler to silence cleanup errors
    # This runs BEFORE TemporaryDirectory cleanup tries to use os.unlink
    def _silent_cleanup() -> None:
        """Restore os functions for cleanup, suppressing any errors."""
        try:
            os.unlink = _original_unlink  # type: ignore[assignment]
            os.rmdir = _original_rmdir  # type: ignore[assignment]
        except Exception:
            pass

    atexit.register(_silent_cleanup)

    # Neutralize dangerous os module functions
    os.kill = None  # type: ignore[assignment]
    os.system = None  # type: ignore[assignment]
    os.putenv = None  # type: ignore[assignment]
    os.remove = None  # type: ignore[assignment]
    os.removedirs = None  # type: ignore[assignment]
    os.rmdir = None  # type: ignore[assignment]
    os.fchdir = None  # type: ignore[assignment]
    os.setuid = None  # type: ignore[assignment]
    os.fork = None  # type: ignore[assignment]
    os.forkpty = None  # type: ignore[assignment]
    os.killpg = None  # type: ignore[assignment]
    os.rename = None  # type: ignore[assignment]
    os.renames = None  # type: ignore[assignment]
    os.truncate = None  # type: ignore[assignment]
    os.replace = None  # type: ignore[assignment]
    os.unlink = None  # type: ignore[assignment]
    os.fchmod = None  # type: ignore[assignment]
    os.fchown = None  # type: ignore[assignment]
    os.chmod = None  # type: ignore[assignment]
    os.chown = None  # type: ignore[assignment]
    os.chroot = None  # type: ignore[assignment]
    os.lchown = None  # type: ignore[assignment]

    # Neutralize dangerous shutil operations
    import shutil

    shutil.rmtree = None  # type: ignore[assignment]
    shutil.move = None  # type: ignore[assignment]
    shutil.chown = None  # type: ignore[assignment]

    # Prevent subprocess usage
    import subprocess

    subprocess.Popen = None  # type: ignore[misc]

    # Block dangerous imports by inserting import hooks

    # Store original __import__
    original_import = builtins.__import__

    def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Import hook that blocks dangerous modules."""
        blocked_modules = {
            "ipdb",
            "joblib",
            "resource",
            "psutil",
            "tkinter",
            "multiprocessing",
            "threading",
            "socket",
            "http",
            "urllib",
            "ftplib",
            "telnetlib",
            "asyncio",
        }
        if name in blocked_modules or name.split(".")[0] in blocked_modules:
            raise ImportError(f"Module '{name}' is not allowed in sandboxed execution")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = _safe_import


def unsafe_execute(code: str, timeout: float, result: list[dict[str, Any]]) -> None:
    """Execute code in isolated context with safety guards.

    This function runs in a separate process and should not be called directly.
    Use execute_code() instead.

    Args:
        code: Python code string to execute
        timeout: Maximum execution time in seconds
        result: Shared list to store execution result
    """
    # Suppress all stderr output from this child process to reduce noise
    # This catches Python internal errors (thread crashes, atexit failures, etc.)
    _suppress_child_process_stderr()

    # Create tempdir in parent scope to ensure cleanup works
    import os
    import shutil
    import tempfile

    # Save original rmtree before reliability_guard disables it
    _original_rmtree = shutil.rmtree

    tmpdir = tempfile.mkdtemp()
    old_cwd = os.getcwd()

    try:
        os.chdir(tmpdir)

        # Apply security restrictions AFTER changing directory
        reliability_guard()

        # Suppress all IO
        with swallow_io():
            try:
                with time_limit(timeout):
                    # Execute code in isolated namespace
                    exec_globals: dict[str, Any] = {}
                    exec(code, exec_globals)
                result.append({"status": "success", "error": None})
            except TimeoutException:
                result.append({"status": "timeout", "error": "Execution timed out"})
            except SyntaxError as e:
                result.append({"status": "syntax_error", "error": str(e)})
            except Exception as e:
                result.append({"status": "runtime_error", "error": f"{type(e).__name__}: {e}"})
    finally:
        # Clean up using saved original function
        try:
            os.chdir(old_cwd)
            _original_rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass  # Best effort cleanup


def _suppress_child_process_stderr() -> None:
    """Redirect stderr to /dev/null to suppress expected sandbox cleanup noise.

    This prevents Python internal error messages from cluttering parent logs:
    - Thread crash tracebacks (EOFError in logging monitor)
    - atexit callback failures (TemporaryDirectory cleanup)
    - Resource cleanup warnings

    The actual execution results are returned via the result list, not stderr.
    """
    try:
        # Redirect stderr to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)  # stderr = file descriptor 2
        os.close(devnull)
    except Exception:
        pass  # Best effort - continue even if this fails


def execute_code(code: str, timeout: float = 5.0) -> dict[str, Any]:
    """Execute Python code in a sandboxed subprocess.

    Args:
        code: Python code string to execute
        timeout: Maximum execution time in seconds (default: 5.0)

    Returns:
        Dictionary with:
            - status: 'success', 'timeout', 'syntax_error', 'runtime_error', 'process_error'
            - error: Error message if status is not 'success', None otherwise

    Example:
        >>> result = execute_code("print('hello')", timeout=1.0)
        >>> result['status']
        'success'
        >>> result = execute_code("while True: pass", timeout=1.0)
        >>> result['status']
        'timeout'
    """
    manager = multiprocessing.Manager()
    result = manager.list()

    try:
        # Spawn separate process for isolation
        process = multiprocessing.Process(target=unsafe_execute, args=(code, timeout, result))
        process.start()

        # Wait for completion with grace period
        process.join(timeout=timeout + 1.0)

        # Force kill if still alive
        if process.is_alive():
            process.kill()
            process.join()
            return {"status": "timeout", "error": "Execution timed out"}

        # Check if process crashed or was killed
        if not result:
            return {"status": "process_error", "error": "Process terminated unexpectedly"}

        return dict(result[0])
    finally:
        # Always shutdown manager to release file descriptors and server process
        # Suppress any cleanup errors - they are harmless but noisy
        try:
            manager.shutdown()
        except Exception:
            pass  # Best effort cleanup - may fail if already in bad state


def check_code_executes(code: str, test_cases: list[str], timeout: float = 5.0) -> dict[str, Any]:
    """Execute code with test cases and return results.

    Args:
        code: Python code string (function definitions)
        test_cases: List of assertion strings to test the code
        timeout: Maximum execution time per test in seconds

    Returns:
        Dictionary with:
            - passed: Number of tests passed
            - total: Total number of tests
            - status: 'all_passed', 'some_failed', 'syntax_error', 'timeout', 'error'
            - error: Error message if applicable
            - test_results: List of per-test results
    """
    # First check if code has syntax errors
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        return {
            "passed": 0,
            "total": len(test_cases),
            "status": "syntax_error",
            "error": str(e),
            "test_results": [],
        }

    test_results = []
    passed = 0
    timeout_count = 0

    for i, test_case in enumerate(test_cases):
        # Combine code with test case
        full_code = f"{code}\n\n{test_case}"

        result = execute_code(full_code, timeout=timeout)

        if result["status"] == "success":
            passed += 1
            test_results.append({"test_idx": i, "passed": True, "error": None})
        else:
            error_msg = result.get("error", "Unknown error")
            test_results.append(
                {
                    "test_idx": i,
                    "passed": False,
                    "error": error_msg,
                }
            )
            # Track timeout occurrences
            if result["status"] == "timeout":
                timeout_count += 1

    # Determine overall status
    if passed == len(test_cases):
        status = "all_passed"
    elif passed > 0:
        status = "some_failed"
    else:
        # All tests failed - check if primarily due to timeout
        # FIX: Use status field instead of string prefix matching on error message
        if timeout_count == len(test_cases):
            status = "timeout"
        else:
            status = "error"

    return {
        "passed": passed,
        "total": len(test_cases),
        "status": status,
        "error": None if status == "all_passed" else f"Passed {passed}/{len(test_cases)} tests",
        "test_results": test_results,
    }
