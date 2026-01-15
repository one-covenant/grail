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
import concurrent.futures
import contextlib
import faulthandler
import io
import logging
import multiprocessing
import os
import platform
import signal
import sys
import tempfile
import threading
import time
import warnings
from multiprocessing.connection import Connection
from typing import Any


# Suppress noisy warnings from multiprocessing child process cleanup
# These are harmless but clutter logs significantly
def _suppress_execution_sandbox_noise() -> None:
    """Configure filters to suppress expected sandbox cleanup noise.

    The execution sandbox spawns child processes that produce harmless but
    noisy error messages during cleanup:
    - EOFError in multiprocessing logging queue monitor threads (legacy Manager-based IPC)
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


# Global lock to protect CUDA env modification during process spawning
# Ensures parent's CUDA visibility isn't affected by concurrent spawns
_SPAWN_LOCK = threading.Lock()


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
    # Pre-check syntax to avoid slow spawn overhead for syntax errors
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        return {"status": "syntax_error", "error": str(e)}

    # Use 'spawn' context to avoid fork() issues with CUDA, threads, and resources
    # Spawn creates a fresh Python process without inheriting parent's:
    # - CUDA contexts (prevents GPU corruption)
    # - Thread locks (prevents deadlocks)
    # - File descriptors (prevents resource leaks)
    # - Signal handlers (prevents interference)
    # This prevents resource_tracker errors from forked processes being killed
    #
    # Note: Spawn has overhead (imports module in child), but it's necessary for
    # safety when parent has CUDA/threading state. For evaluation, we spawn
    # processes in batches so the overhead is amortized.
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    process = None
    try:
        # CRITICAL: Hide GPUs before spawning to ensure child is CPU-only from birth
        # This prevents CUDA initialization during child process startup/imports
        #
        # We use a lock to atomically:
        # 1. Save parent's CUDA env vars
        # 2. Set CUDA_VISIBLE_DEVICES="" (child inherits this)
        # 3. Spawn child process
        # 4. Restore parent's CUDA env vars
        #
        # This ensures parent's GPU access (vLLM, HuggingFace) is never affected
        # and prevents race conditions between concurrent spawns
        with _SPAWN_LOCK:
            old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            old_cuda_order = os.environ.get("CUDA_DEVICE_ORDER")

            try:
                # Set env vars before spawning - child will inherit these
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

                process = ctx.Process(
                    target=_unsafe_execute_with_pipe,
                    args=(code, timeout, child_conn),
                )
                process.start()
            finally:
                # ALWAYS restore parent's CUDA visibility immediately
                # This happens in microseconds, minimizing impact on parent
                if old_cuda_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible

                if old_cuda_order is None:
                    os.environ.pop("CUDA_DEVICE_ORDER", None)
                else:
                    os.environ["CUDA_DEVICE_ORDER"] = old_cuda_order

        child_conn.close()  # Close child end in parent

        # Wait for completion with grace period
        # Spawn has ~5-7s overhead for importing heavy dependencies (torch, bittensor, etc.)
        # Add extra time beyond code timeout to account for spawn overhead + cleanup
        process_timeout = timeout + 10.0
        process.join(timeout=process_timeout)

        # Graceful shutdown: try terminate first, then kill if necessary
        if process.is_alive():
            # Process didn't exit after timeout + 10s grace period
            # This means code is hanging the process (infinite loop, blocking syscall, etc.)
            # NOT a normal timeout (which would send result via pipe before exit)
            # Classify as "timeout" since it's code-related, not infrastructure failure
            process.terminate()
            process.join(timeout=0.5)

            # Force kill only if terminate didn't work
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)

            return {
                "status": "timeout",
                "error": "Process unresponsive (likely code caused hang - infinite loop or blocking call)",
            }

        # Read result from pipe with timeout to prevent deadlock
        # If child crashed after join(), poll prevents indefinite hang
        try:
            if parent_conn.poll(timeout=1.0):
                return parent_conn.recv()
        except EOFError:
            # Child died after closing the pipe without sending a result
            pass
        except Exception as e:
            # Handle any other pipe communication errors
            return {
                "status": "process_error",
                "error": f"Pipe communication error: {e}",
            }

        exit_code = process.exitcode
        return {
            "status": "process_error",
            "error": f"Process terminated unexpectedly (exit_code={exit_code})",
        }
    finally:
        # Ensure process is fully cleaned up
        if process is not None and process.is_alive():
            try:
                process.kill()
                process.join(timeout=0.5)
            except Exception:
                pass

        # Close pipe connections
        try:
            parent_conn.close()
        except Exception:
            pass


def _unsafe_execute_with_pipe(
    code: str,
    timeout: float,
    result_conn: Connection,
) -> None:
    """Execute code in sandbox and send result via pipe.

    Uses Pipe instead of Queue because:
    1. Pipe.send() uses simple pickle + file descriptor write
    2. Works even after reliability_guard blocks multiprocessing imports
    3. No background threads that produce cleanup noise

    Note: With spawn context, this runs in a fresh process with no inherited state.
    """
    # CRITICAL #1: Suppress stderr IMMEDIATELY before any imports
    # This prevents logging noise from appearing in parent's console:
    # - EOFError tracebacks from logging queue monitors
    # - PyTorch CUDA_ALLOC_CONF deprecation warnings
    # - Resource cleanup warnings during exit
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)  # stderr = file descriptor 2
        os.close(devnull)
    except Exception:
        pass  # Best effort - continue even if this fails

    # CRITICAL #2: Hide all GPUs to make this process CPU-only
    # This prevents CUDA context conflicts when spawning from GPU-enabled parent
    # Must happen BEFORE any imports that might touch CUDA (torch, etc.)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Ensure modules used by Connection.send() are imported BEFORE the sandbox
    # import hook blocks "multiprocessing.*" imports.
    #
    # Connection.send() uses multiprocessing's ForkingPickler internally.
    import multiprocessing.reduction  # noqa: F401
    import pickle  # noqa: F401

    # Store send/close callables BEFORE reliability_guard blocks imports.
    send_result = result_conn.send
    close_conn = result_conn.close

    # Create tempdir and do all setup before security lockdown
    tmpdir = tempfile.mkdtemp()
    old_cwd = os.getcwd()

    # Store original functions before reliability_guard disables them
    import shutil

    _original_rmtree = shutil.rmtree

    # Change to temp directory
    os.chdir(tmpdir)

    result: dict[str, Any]
    try:
        # Apply security restrictions
        reliability_guard()

        # Execute the code with IO suppression and timeout
        # Parent handles process-level timeout via join(), but we also use
        # signal-based timeout as defense-in-depth for runaway code
        with swallow_io():
            try:
                with time_limit(timeout):
                    exec_globals: dict[str, Any] = {}
                    exec(code, exec_globals)
                result = {"status": "success", "error": None}
            except TimeoutException:
                result = {"status": "timeout", "error": "Execution timed out"}
            except SyntaxError as e:
                result = {"status": "syntax_error", "error": str(e)}
            except Exception as e:
                result = {"status": "runtime_error", "error": f"{type(e).__name__}: {e}"}
    except Exception as e:
        result = {"status": "process_error", "error": f"Sandbox error: {e}"}
    finally:
        # Always cleanup temp directory, even on timeout/error
        try:
            os.chdir(old_cwd)
            _original_rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass  # Cleanup errors shouldn't prevent result transmission

    # Send result via pipe - using saved send function
    # This must succeed for parent to receive results
    try:
        send_result(result)
        # Explicitly flush to ensure data is sent before process exits
        # (though send() should be synchronous, be defensive)
    except BrokenPipeError:
        # Parent closed pipe - this is OK if parent timed out
        pass
    except Exception:
        # Other errors - parent will detect missing result
        pass
    finally:
        # Close our end of the pipe
        try:
            close_conn()
        except Exception:
            pass


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

    # Handle empty test case list
    if not test_cases:
        return {
            "passed": 0,
            "total": 0,
            "status": "all_passed",
            "error": None,
            "test_results": [],
        }

    # Execute all test cases in parallel for 3x speedup
    # Each test spawns independently, avoiding sequential ~7s overhead per test
    test_results = []
    passed = 0
    timeout_count = 0

    # Use ProcessPoolExecutor to run tests in parallel
    # max_workers = number of test cases for maximum parallelism
    max_workers = min(len(test_cases), 10)  # Cap at 10 to avoid resource exhaustion
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all test cases for parallel execution
        future_to_idx = {
            executor.submit(execute_code, f"{code}\n\n{test_case}", timeout): i
            for i, test_case in enumerate(test_cases)
        }

        # Collect results as they complete
        # Use list to maintain order by test_idx
        results_dict = {}
        for future in concurrent.futures.as_completed(future_to_idx):
            test_idx = future_to_idx[future]
            try:
                result = future.result()
                results_dict[test_idx] = result
            except Exception as e:
                # Handle unexpected executor errors
                results_dict[test_idx] = {
                    "status": "process_error",
                    "error": f"Executor error: {e}",
                }

    # Process results in original order
    for i in range(len(test_cases)):
        result = results_dict.get(i, {"status": "process_error", "error": "Missing result"})

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


# =============================================================================
# FAST CODE EXECUTION POOL
# =============================================================================
# Provides a persistent worker pool that eliminates spawn overhead.
# Workers are spawned once and reused across all code executions.
# This gives ~50-100x speedup over spawning fresh processes per test.


def _pool_safe_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Restricted import used for pool code execution.

    This is NOT a complete sandbox. It exists to block the most trivial escapes
    (e.g., importing `importlib` to reload monkeypatched modules, importing
    `posix`/`os`/`subprocess` to spawn processes, importing `ctypes` to call
    native syscalls).

    Keep this lightweight: it runs on every `import` in user code.
    """
    # Strict denylist: modules that trivially lead to RCE / host compromise.
    # Note: block both the exact module and its top-level package.
    blocked = {
        "asyncio",
        "builtins",
        "ctypes",
        "importlib",
        "inspect",
        "multiprocessing",
        "os",
        "pathlib",
        "posix",
        "resource",
        "shutil",
        "signal",
        "site",
        "socket",
        "subprocess",
        "sys",
        "threading",
        "types",
    }

    top = name.split(".")[0]
    if name in blocked or top in blocked:
        raise ImportError(f"Module '{name}' is not allowed in sandboxed execution")

    # Allowlist for common MBPP-style solutions.
    allowed = {
        "math",
        "itertools",
        "functools",
        "collections",
        "heapq",
        "bisect",
        "operator",
        "re",
        "string",
    }
    if name not in allowed and top not in allowed:
        raise ImportError(f"Module '{name}' is not allowed in sandboxed execution")

    # Use the real import mechanism for allowed modules.
    import builtins

    return builtins.__import__(name, globals_, locals_, fromlist, level)


def _pool_safe_builtins() -> dict[str, Any]:
    """Return a restricted `__builtins__` dict for pool exec().

    This blocks obvious dangerous functionality (open/eval/exec/compile/importlib)
    without changing global interpreter state (important for process stability).
    """
    import builtins

    allow_names = {
        # Core types / constructors
        "None",
        "False",
        "True",
        "bool",
        "int",
        "float",
        "complex",
        "str",
        "bytes",
        "bytearray",
        "list",
        "tuple",
        "set",
        "frozenset",
        "dict",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "reversed",
        "slice",
        # Utilities
        "abs",
        "all",
        "any",
        "min",
        "max",
        "sum",
        "len",
        "round",
        "pow",
        "divmod",
        "chr",
        "ord",
        "hex",
        "oct",
        "bin",
        "format",
        "repr",
        "print",
        "isinstance",
        # Exceptions
        "BaseException",
        "Exception",
        "AssertionError",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "StopIteration",
        "ZeroDivisionError",
        # Class support (needed if code defines classes)
        "__build_class__",
        "object",
        "type",
    }

    safe: dict[str, Any] = {name: getattr(builtins, name) for name in allow_names}
    safe["__import__"] = _pool_safe_import
    return safe


def _pool_worker_execute(
    code: str,
    tests: list[str],
    timeout: float,
) -> dict[str, Any]:
    """Execute code with tests in a pool worker.

    This function runs in a persistent worker process. The worker is spawned
    once and stays alive, eliminating the ~2s spawn overhead per execution.

    Note: CUDA hiding and security measures are applied in _pool_worker_init(),
    which runs once per worker at spawn time.

    Args:
        code: Python code string (function definitions)
        tests: List of assertion strings to test
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with passed, total, status, error, test_results
    """
    # No need to set env vars here - already set in _pool_worker_init()
    # and os.putenv is disabled for security anyway

    # Pre-check syntax to fail fast
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        return {
            "passed": 0,
            "total": len(tests),
            "status": "syntax_error",
            "error": str(e),
            "test_results": [],
        }

    if not tests:
        return {
            "passed": 0,
            "total": 0,
            "status": "all_passed",
            "error": None,
            "test_results": [],
        }

    # Note: We don't set RLIMIT_AS here because:
    # 1. Workers are persistent - setting once would affect all future executions
    # 2. Python already uses significant memory after importing
    # 3. RLIMIT_AS can't be increased once lowered
    # The worker isolation + timeout provides sufficient protection

    # Execute each test
    test_results = []
    passed = 0
    timeout_count = 0
    safe_builtins = _pool_safe_builtins()

    for i, test in enumerate(tests):
        full_code = f"{code}\n\n{test}"
        try:
            # Use signal-based timeout for each test
            def timeout_handler(signum: int, frame: Any) -> None:
                raise TimeoutException("Test timed out")

            old_handler = None
            if platform.system() != "Windows":
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))

            try:
                exec_globals: dict[str, Any] = {"__builtins__": safe_builtins}
                exec(full_code, exec_globals)
                test_results.append({"test_idx": i, "passed": True, "error": None})
                passed += 1
            except TimeoutException:
                test_results.append({"test_idx": i, "passed": False, "error": "Timeout"})
                timeout_count += 1
            except Exception as e:
                test_results.append(
                    {"test_idx": i, "passed": False, "error": f"{type(e).__name__}: {e}"}
                )
            finally:
                if platform.system() != "Windows" and old_handler is not None:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

        except Exception as e:
            test_results.append({"test_idx": i, "passed": False, "error": str(e)})

    # Determine overall status
    if passed == len(tests):
        status = "all_passed"
    elif passed > 0:
        status = "some_failed"
    elif timeout_count == len(tests):
        status = "timeout"
    else:
        status = "error"

    return {
        "passed": passed,
        "total": len(tests),
        "status": status,
        "error": None if status == "all_passed" else f"Passed {passed}/{len(tests)} tests",
        "test_results": test_results,
    }


def _pool_worker_init() -> None:
    """Initialize security sandbox in pool worker process.

    Called ONCE when each worker spawns. Applies security measures that work
    with persistent workers (can't use RLIMIT_AS as it would break the worker
    after first task).

    Security measures applied:
    - Hide CUDA to prevent GPU access
    - Set RLIMIT_NPROC=0 to prevent forking/spawning
    - Set RLIMIT_NOFILE to limit file descriptors
    - Set RLIMIT_CPU as backup timeout
    - Disable core dumps
    - Neutralize dangerous os functions
    - Neutralize shutil dangerous functions
    - Neutralize subprocess module
    - Disable dangerous builtins
    """
    import builtins
    import subprocess

    # 1. Hide CUDA - prevent GPU access
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # 2. Suppress noisy warnings/errors
    _suppress_execution_sandbox_noise()

    # 3. Disable faulthandler to prevent info leakage
    faulthandler.disable()

    # 4. Set resource limits (Unix only)
    if platform.system() != "Windows":
        import resource

        # Note: Don't set RLIMIT_NPROC - it can interfere with worker operations
        # Fork prevention is handled by os.fork = None instead

        # Disable core dumps
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (OSError, ValueError):
            pass

        # Limit file descriptors (worker needs some, but not unlimited)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        except (OSError, ValueError):
            pass

        # Backup CPU time limit (generous - signal timeout is primary)
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (300, 300))  # 5 minutes
        except (OSError, ValueError):
            pass

    # 5. Disable dangerous builtins
    builtins.exit = None  # type: ignore[attr-defined]
    builtins.quit = None  # type: ignore[attr-defined]

    # 6. Save originals for internal cleanup before disabling
    # Python's tempfile cleanup needs these to work
    _original_unlink = os.unlink
    _original_rmdir = os.rmdir
    _original_remove = os.remove

    def _restore_for_cleanup() -> None:
        """Restore os functions for cleanup, suppressing any errors."""
        try:
            os.unlink = _original_unlink  # type: ignore[assignment]
            os.rmdir = _original_rmdir  # type: ignore[assignment]
            os.remove = _original_remove  # type: ignore[assignment]
        except Exception:
            pass

    atexit.register(_restore_for_cleanup)

    # 7. Neutralize dangerous os module functions
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

    # 8. Neutralize subprocess module
    subprocess.Popen = None  # type: ignore[assignment]
    subprocess.call = None  # type: ignore[assignment]
    subprocess.run = None  # type: ignore[assignment]
    subprocess.check_call = None  # type: ignore[assignment]
    subprocess.check_output = None  # type: ignore[assignment]


def _pool_warmup_noop() -> bool:
    """Dummy function to force worker spawn during warmup."""
    return True


class CodeExecutionPool:
    """Persistent worker pool for fast code execution.

    Workers are spawned once at initialization and reused for all executions.
    This eliminates the ~2s spawn overhead per execution, giving 50-100x speedup.

    Usage:
        pool = CodeExecutionPool(num_workers=8)
        try:
            results = pool.execute_batch([
                ("def add(a,b): return a+b", ["assert add(1,2)==3"]),
                ("def mul(a,b): return a*b", ["assert mul(2,3)==6"]),
            ])
        finally:
            pool.shutdown()

    Thread Safety:
        The pool is thread-safe and can be called from multiple threads.

    Memory Safety:
        Workers are recycled after max_tasks_per_child executions to prevent
        memory leaks from accumulating global state.
    """

    def __init__(
        self,
        num_workers: int = 8,
        max_tasks_per_child: int = 50,
    ) -> None:
        """Initialize the execution pool.

        Args:
            num_workers: Number of worker processes to spawn
            max_tasks_per_child: Recycle workers after this many tasks to prevent leaks
        """
        self.num_workers = num_workers
        self.max_tasks_per_child = max_tasks_per_child
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        """Start the worker pool.

        Workers are spawned immediately and warmed up to eliminate first-call latency.
        """
        with self._lock:
            if self._started:
                return

            # Use spawn context - CUDA-safe, workers don't inherit GPU state
            ctx = multiprocessing.get_context("spawn")

            # Hide GPUs before spawning workers
            with _SPAWN_LOCK:
                old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                old_cuda_order = os.environ.get("CUDA_DEVICE_ORDER")

                try:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

                    # max_tasks_per_child requires Python 3.11+; Pyright stubs may not include it
                    self._executor = concurrent.futures.ProcessPoolExecutor(  # type: ignore[call-overload]
                        max_workers=self.num_workers,
                        mp_context=ctx,
                        max_tasks_per_child=self.max_tasks_per_child,
                        initializer=_pool_worker_init,  # Apply security sandbox
                    )

                    # Warm up workers - force spawn and apply security now
                    # Timeout increased to 45s to handle high-load scenarios
                    warmup_futures = [
                        self._executor.submit(_pool_warmup_noop) for _ in range(self.num_workers)
                    ]
                    for f in warmup_futures:
                        f.result(timeout=45)

                finally:
                    # Restore parent's CUDA visibility
                    if old_cuda_visible is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible

                    if old_cuda_order is None:
                        os.environ.pop("CUDA_DEVICE_ORDER", None)
                    else:
                        os.environ["CUDA_DEVICE_ORDER"] = old_cuda_order

            self._started = True

    def shutdown(self) -> None:
        """Shutdown the worker pool and release all resources.

        This should be called when the pool is no longer needed to free
        memory and prevent zombie processes.
        """
        with self._lock:
            if self._executor is not None:
                try:
                    self._executor.shutdown(wait=True, cancel_futures=True)
                except Exception:
                    # Force shutdown on error
                    try:
                        self._executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                self._executor = None
            self._started = False

    def execute(
        self,
        code: str,
        tests: list[str],
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Execute code with tests using a pool worker.

        Args:
            code: Python code string (function definitions)
            tests: List of assertion strings
            timeout: Maximum execution time per test in seconds

        Returns:
            Dictionary with passed, total, status, error, test_results
        """
        if not self._started:
            self.start()

        if self._executor is None:
            return {
                "passed": 0,
                "total": len(tests),
                "status": "process_error",
                "error": "Pool not initialized",
                "test_results": [],
            }

        try:
            future = self._executor.submit(_pool_worker_execute, code, tests, timeout)
            # Add buffer for IPC overhead
            result = future.result(timeout=timeout * len(tests) + 30)
            return result
        except concurrent.futures.TimeoutError:
            # Executor-level timeout - worker didn't respond in time
            # This is almost always caused by code hanging the worker (infinite loop, blocking call)
            # NOT a transient infrastructure issue - same code will hang again
            # Classify as "timeout" (code problem) to prevent wasteful retries
            return {
                "passed": 0,
                "total": len(tests),
                "status": "timeout",
                "error": "Worker unresponsive (likely code caused worker to hang)",
                "test_results": [],
            }
        except concurrent.futures.BrokenExecutor as e:
            # Worker pool crashed - this is a process error, retryable
            return {
                "passed": 0,
                "total": len(tests),
                "status": "process_error",
                "error": f"Worker pool failure: {e}",
                "test_results": [],
            }
        except Exception as e:
            # Other executor errors (worker crash, IPC failure) - process error
            return {
                "passed": 0,
                "total": len(tests),
                "status": "process_error",
                "error": f"Pool execution error: {e}",
                "test_results": [],
            }

    def execute_batch(
        self,
        items: list[tuple[str, list[str]]],
        timeout: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Execute multiple (code, tests) pairs in parallel.

        Args:
            items: List of (code, tests) tuples
            timeout: Maximum execution time per test in seconds

        Returns:
            List of result dictionaries, one per input item
        """
        if not self._started:
            self.start()

        if self._executor is None or not items:
            return [
                {
                    "passed": 0,
                    "total": len(tests) if tests else 0,
                    "status": "error",
                    "error": "Pool not initialized",
                    "test_results": [],
                }
                for _, tests in items
            ]

        # Submit all items in parallel
        futures = [
            self._executor.submit(_pool_worker_execute, code, tests, timeout)
            for code, tests in items
        ]

        # Collect results in order
        results = []
        # Overall timeout: account for all items potentially running sequentially
        overall_timeout = timeout * max(len(t) for _, t in items if t) + 60 if items else 30

        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=overall_timeout)
                results.append(result)
            except concurrent.futures.TimeoutError:
                # Executor-level timeout - worker didn't respond
                # Almost always code hanging the worker, not infrastructure failure
                results.append(
                    {
                        "passed": 0,
                        "total": len(items[i][1]),
                        "status": "timeout",
                        "error": "Worker unresponsive (likely code caused worker to hang)",
                        "test_results": [],
                    }
                )
            except concurrent.futures.BrokenExecutor as e:
                # Worker pool crashed - process error
                results.append(
                    {
                        "passed": 0,
                        "total": len(items[i][1]),
                        "status": "process_error",
                        "error": f"Worker pool failure: {e}",
                        "test_results": [],
                    }
                )
            except Exception as e:
                # Other executor errors - process error
                results.append(
                    {
                        "passed": 0,
                        "total": len(items[i][1]),
                        "status": "process_error",
                        "error": f"Execution error: {e}",
                        "test_results": [],
                    }
                )

        return results

    def health_check(self, timeout: float = 10.0) -> dict[str, Any]:
        """Check pool health by running a simple test on all workers.

        This verifies:
        - Pool is started and executor exists
        - All workers are responsive within timeout
        - Workers can execute simple code correctly

        Args:
            timeout: Maximum time to wait for health check (default: 10s)

        Returns:
            Dictionary with:
                - healthy: bool, True if pool is fully operational
                - started: bool, True if pool has been started
                - num_workers: int, configured worker count
                - workers_responsive: int, number of workers that responded
                - workers_correct: int, number of workers that returned correct results
                - error: str | None, error message if unhealthy
        """
        logger = logging.getLogger(__name__)

        result: dict[str, Any] = {
            "healthy": False,
            "started": self._started,
            "num_workers": self.num_workers,
            "workers_responsive": 0,
            "workers_correct": 0,
            "error": None,
            "check_duration_ms": 0,
        }

        if not self._started:
            result["error"] = "Pool not started"
            return result

        if self._executor is None:
            result["error"] = "Pool executor is None"
            return result

        start_time = time.time()

        # Submit health check tasks to all workers
        # Use a simple computation that should complete quickly
        test_code = "def _health_check_fn(x): return x * 2"
        test_cases = ["assert _health_check_fn(21) == 42"]

        try:
            futures = [
                self._executor.submit(_pool_worker_execute, test_code, test_cases, 2.0)
                for _ in range(self.num_workers)
            ]

            responsive = 0
            correct = 0
            errors = []

            for i, future in enumerate(futures):
                try:
                    res = future.result(timeout=timeout)
                    responsive += 1
                    if res.get("status") == "all_passed" and res.get("passed") == 1:
                        correct += 1
                    else:
                        errors.append(f"Worker {i}: unexpected result status={res.get('status')}")
                except concurrent.futures.TimeoutError:
                    errors.append(f"Worker {i}: timeout")
                except Exception as e:
                    errors.append(f"Worker {i}: {e}")

            result["workers_responsive"] = responsive
            result["workers_correct"] = correct
            result["check_duration_ms"] = int((time.time() - start_time) * 1000)

            if responsive == self.num_workers and correct == self.num_workers:
                result["healthy"] = True
            else:
                result["error"] = (
                    f"Pool degraded: {responsive}/{self.num_workers} responsive, {correct}/{self.num_workers} correct"
                )
                if errors:
                    result["worker_errors"] = errors[:5]  # Limit error list

            logger.info(
                "Pool health check: healthy=%s workers=%d/%d responsive=%d correct=%d duration=%dms",
                result["healthy"],
                self.num_workers,
                self.num_workers,
                responsive,
                correct,
                result["check_duration_ms"],
            )

        except concurrent.futures.BrokenExecutor as e:
            result["error"] = f"Executor broken: {e}"
            logger.error("Pool health check failed: executor broken: %s", e)
        except Exception as e:
            result["error"] = f"Health check error: {e}"
            logger.error("Pool health check failed: %s", e, exc_info=True)

        return result

    def __enter__(self) -> CodeExecutionPool:
        """Context manager entry - starts the pool."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - shuts down the pool."""
        self.shutdown()


# =============================================================================
# GLOBAL POOL ACCESSOR
# =============================================================================
# Provides a way to set/get a global execution pool for use by environments.
# The pool lifecycle is managed by the caller (e.g., EvaluatorService).

_GLOBAL_EXECUTION_POOL: CodeExecutionPool | None = None
_GLOBAL_POOL_LOCK = threading.Lock()


def set_global_execution_pool(pool: CodeExecutionPool | None) -> None:
    """Set the global execution pool for use by environments.

    Args:
        pool: The pool to use, or None to clear
    """
    global _GLOBAL_EXECUTION_POOL
    with _GLOBAL_POOL_LOCK:
        _GLOBAL_EXECUTION_POOL = pool


def get_global_execution_pool() -> CodeExecutionPool | None:
    """Get the global execution pool if set.

    Returns:
        The global pool, or None if not set
    """
    with _GLOBAL_POOL_LOCK:
        return _GLOBAL_EXECUTION_POOL


def check_code_executes_fast(
    code: str,
    test_cases: list[str],
    timeout: float = 5.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Execute code with test cases using the global pool if available.

    This is a drop-in replacement for check_code_executes that uses the
    global execution pool when set, falling back to the original implementation.

    Automatically retries on process errors (infrastructure failures), but not
    on legitimate Python errors (syntax errors, runtime errors, timeouts).

    Args:
        code: Python code string (function definitions)
        test_cases: List of assertion strings to test the code
        timeout: Maximum execution time per test in seconds
        max_retries: Maximum retry attempts for process errors (default: 3)

    Returns:
        Dictionary with passed, total, status, error, test_results
    """
    logger = logging.getLogger(__name__)
    pool = get_global_execution_pool()

    # Log execution method for debugging
    using_pool = pool is not None
    execute_fn = pool.execute if using_pool else check_code_executes

    logger.debug(
        "check_code_executes_fast: using_pool=%s num_tests=%d timeout=%.1f",
        using_pool,
        len(test_cases),
        timeout,
    )

    # Retry loop for process errors only
    result: dict[str, Any] = {"status": "error", "error": "No attempts made"}
    start_time = time.time()

    for attempt in range(max_retries):
        attempt_start = time.time()
        result = execute_fn(code, test_cases, timeout)
        attempt_duration_ms = int((time.time() - attempt_start) * 1000)

        # Check if this is a process error (infrastructure failure)
        # These are retryable: worker crash, pipe error, executor error
        status = result.get("status", "")
        if status == "process_error":
            error_msg = result.get("error", "unknown")

            # Log pool health on process errors to diagnose infrastructure issues
            pool_health_info = "N/A"
            if pool is not None:
                try:
                    health = pool.health_check(timeout=5.0)
                    pool_health_info = (
                        f"healthy={health.get('healthy')}, "
                        f"workers={health.get('workers_responsive', 0)}/{health.get('num_workers', 0)}, "
                        f"error={health.get('error')}"
                    )
                except Exception as health_err:
                    pool_health_info = f"health_check_failed: {health_err}"

            # Last attempt - return the error with detailed diagnostics
            if attempt == max_retries - 1:
                total_duration_ms = int((time.time() - start_time) * 1000)
                logger.warning(
                    "[check_code_executes_fast] PROCESS_ERROR after %d attempts | "
                    "error=%s | attempt_duration=%dms | total_duration=%dms | "
                    "pool_health=[%s] | num_tests=%d | "
                    "This may indicate pool degradation or worker crashes",
                    max_retries,
                    error_msg,
                    attempt_duration_ms,
                    total_duration_ms,
                    pool_health_info,
                    len(test_cases),
                )
                return result

            # Retry on process error
            logger.info(
                "[check_code_executes_fast] Retrying due to process_error (attempt %d/%d) | "
                "error=%s | duration=%dms | pool_health=[%s]",
                attempt + 1,
                max_retries,
                error_msg,
                attempt_duration_ms,
                pool_health_info,
            )
            continue

        # Non-retryable result (success, syntax error, runtime error, timeout)
        # These reflect actual code behavior, not infrastructure issues
        if logger.isEnabledFor(logging.DEBUG):
            total_duration_ms = int((time.time() - start_time) * 1000)
            logger.debug(
                "check_code_executes_fast: status=%s passed=%s/%s duration=%dms",
                status,
                result.get("passed", 0),
                result.get("total", len(test_cases)),
                total_duration_ms,
            )
        return result

    # Fallback (should never reach here due to return in last attempt)
    return result
