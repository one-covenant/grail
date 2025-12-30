"""Unit tests for safe code execution sandbox.

Tests the execution.py module's handling of:
- Timeouts and process termination
- Resource cleanup and leak prevention
- Syntax and runtime errors
- Edge cases and multiprocessing safety
"""

from __future__ import annotations

import time

import pytest

from grail.environments.execution import check_code_executes, execute_code


class TestBasicExecution:
    """Test basic code execution functionality."""

    def test_successful_execution(self) -> None:
        """Test simple successful code execution."""
        result = execute_code("x = 1 + 1", timeout=5.0)
        assert result["status"] == "success"
        assert result["error"] is None

    def test_syntax_error_fast_path(self) -> None:
        """Test syntax errors are caught via pre-compilation (fast path)."""
        start = time.time()
        result = execute_code("def broken_func(:", timeout=5.0)
        elapsed = time.time() - start

        assert result["status"] == "syntax_error"
        assert "invalid syntax" in result["error"].lower()
        # Should be instant (no process spawn overhead)
        assert elapsed < 1.0

    def test_runtime_error_captured(self) -> None:
        """Test runtime errors are properly captured."""
        result = execute_code("raise ValueError('test error')", timeout=5.0)
        assert result["status"] == "runtime_error"
        assert "ValueError" in result["error"]
        assert "test error" in result["error"]

    def test_successful_computation(self) -> None:
        """Test execution of actual computation."""
        code = """
result = sum(range(100))
factorial = 1
for i in range(1, 6):
    factorial *= i
"""
        result = execute_code(code, timeout=5.0)
        assert result["status"] == "success"


class TestTimeoutHandling:
    """Test timeout mechanisms and process termination."""

    def test_timeout_enforcement(self) -> None:
        """Test that infinite loops are properly terminated."""
        start = time.time()
        result = execute_code("while True: pass", timeout=1.0)
        elapsed = time.time() - start

        assert result["status"] == "timeout"
        assert "timed out" in result["error"].lower()
        # Should complete within timeout + overhead (spawn ~7s + timeout 1s + grace 10s)
        assert elapsed < 20.0

    def test_sleep_timeout(self) -> None:
        """Test timeout on sleep operation."""
        result = execute_code("import time; time.sleep(100)", timeout=1.0)
        assert result["status"] == "timeout"

    def test_timeout_boundary(self) -> None:
        """Test code that completes just within timeout."""
        # Code that takes ~0.5s but timeout is 2s
        code = """
import time
time.sleep(0.1)  # Well within timeout
result = sum(range(1000))
"""
        result = execute_code(code, timeout=2.0)
        assert result["status"] == "success"


class TestResourceCleanup:
    """Test resource cleanup and leak prevention."""

    def test_multiple_rapid_executions(self) -> None:
        """Test that rapid executions don't leak resources."""
        results = []
        for i in range(5):
            result = execute_code(f"x = {i} * 2", timeout=5.0)
            results.append(result["status"])

        # All should succeed
        assert all(status == "success" for status in results)

    def test_process_cleanup_on_timeout(self) -> None:
        """Test that timed-out processes are properly cleaned up."""
        # Run multiple timeouts to verify cleanup
        for _ in range(3):
            result = execute_code("while True: pass", timeout=0.5)
            assert result["status"] == "timeout"
        # If cleanup fails, we'd see resource_tracker warnings or hangs

    def test_cleanup_after_exception(self) -> None:
        """Test cleanup after runtime exceptions."""
        for _ in range(3):
            result = execute_code("raise RuntimeError('cleanup test')", timeout=5.0)
            assert result["status"] == "runtime_error"


class TestSandboxSecurity:
    """Test sandbox restrictions and security."""

    def test_dangerous_imports_blocked(self) -> None:
        """Test that dangerous module imports are blocked in sandbox."""
        # These modules are blocked by reliability_guard's import hook
        blocked_modules = ["socket", "threading", "multiprocessing"]

        for module in blocked_modules:
            code = f"import {module}"
            result = execute_code(code, timeout=5.0)
            # Import should be blocked with ImportError
            assert result["status"] == "runtime_error", f"Expected error for: import {module}"
            assert "ImportError" in result.get("error", ""), f"Expected ImportError for: {module}"

    def test_dangerous_operations_disabled(self) -> None:
        """Test that dangerous operations are disabled even if imports work."""
        # subprocess import is allowed but Popen is disabled
        code = "import subprocess; subprocess.Popen(['ls'])"
        result = execute_code(code, timeout=5.0)
        # Should fail because subprocess.Popen is set to None
        assert result["status"] == "runtime_error"

    def test_file_operations_restricted(self) -> None:
        """Test that file operations are restricted."""
        code = """
import os
os.remove('/tmp/test')  # Should be blocked
"""
        result = execute_code(code, timeout=5.0)
        # Should fail because os.remove is disabled
        assert result["status"] == "runtime_error"


class TestCheckCodeExecutes:
    """Test the check_code_executes function with test cases."""

    def test_all_tests_pass(self) -> None:
        """Test code where all test cases pass."""
        code = "def add(a, b):\n    return a + b"
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]

        result = check_code_executes(code, tests, timeout=5.0)

        assert result["status"] == "all_passed"
        assert result["passed"] == 3
        assert result["total"] == 3
        assert result["error"] is None

    def test_some_tests_fail(self) -> None:
        """Test code where some test cases fail."""
        code = "def multiply(a, b):\n    return a + b"  # Bug: using + instead of *
        tests = ["assert multiply(2, 3) == 6", "assert multiply(0, 5) == 0"]

        result = check_code_executes(code, tests, timeout=5.0)

        assert result["status"] in ("some_failed", "error")
        assert result["passed"] < result["total"]
        assert result["total"] == 2

    def test_syntax_error_in_code(self) -> None:
        """Test handling of syntax errors in code."""
        code = "def broken_func(:\n    return 42"
        tests = ["assert broken_func() == 42"]

        result = check_code_executes(code, tests, timeout=5.0)

        assert result["status"] == "syntax_error"
        assert result["passed"] == 0
        assert "syntax" in result["error"].lower()

    def test_timeout_in_tests(self) -> None:
        """Test handling of timeouts during test execution."""
        code = "def infinite():\n    while True:\n        pass"
        tests = ["infinite()"]

        result = check_code_executes(code, tests, timeout=0.5)

        assert result["status"] == "timeout"
        assert result["passed"] == 0

    def test_empty_test_list(self) -> None:
        """Test handling of empty test list."""
        code = "def foo():\n    return 42"
        tests: list[str] = []

        result = check_code_executes(code, tests, timeout=5.0)

        assert result["status"] == "all_passed"
        assert result["passed"] == 0
        assert result["total"] == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_code(self) -> None:
        """Test execution of empty code."""
        result = execute_code("", timeout=5.0)
        assert result["status"] == "success"

    def test_code_with_comments_only(self) -> None:
        """Test code with only comments."""
        result = execute_code("# This is a comment\n# Another comment", timeout=5.0)
        assert result["status"] == "success"

    def test_unicode_in_code(self) -> None:
        """Test execution with unicode characters."""
        code = 'message = "Hello, 世界! 🌍"\nresult = len(message)'
        result = execute_code(code, timeout=5.0)
        assert result["status"] == "success"

    def test_large_computation(self) -> None:
        """Test execution of larger computation."""
        code = """
# Fibonacci computation
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

result = fib(10)
"""
        result = execute_code(code, timeout=5.0)
        assert result["status"] == "success"

    def test_multiple_syntax_errors(self) -> None:
        """Test code with multiple syntax errors."""
        code = "def bad( def worse(:"
        result = execute_code(code, timeout=5.0)
        assert result["status"] == "syntax_error"

    def test_indentation_error(self) -> None:
        """Test detection of indentation errors."""
        code = "def foo():\nreturn 42"  # Missing indentation
        result = execute_code(code, timeout=5.0)
        assert result["status"] == "syntax_error"


class TestMultiprocessingIsolation:
    """Test multiprocessing isolation and safety."""

    @pytest.mark.slow
    def test_no_state_leakage(self) -> None:
        """Test that executions don't leak state between processes."""
        # First execution sets a global
        code1 = "GLOBAL_VAR = 42"
        result1 = execute_code(code1, timeout=5.0)
        assert result1["status"] == "success"

        # Second execution should not see GLOBAL_VAR
        code2 = """
try:
    _ = GLOBAL_VAR
    state_leaked = True
except NameError:
    state_leaked = False
assert not state_leaked
"""
        result2 = execute_code(code2, timeout=5.0)
        assert result2["status"] == "success"

    @pytest.mark.slow
    def test_concurrent_safety(self) -> None:
        """Test that multiple executions can run in sequence safely."""
        import concurrent.futures

        def run_execution(seed: int) -> str:
            code = f"x = {seed} * 2"
            result = execute_code(code, timeout=5.0)
            return result["status"]

        # Run multiple executions sequentially (not truly concurrent)
        # to verify process pool doesn't interfere
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(run_execution, i) for i in range(3)]
            results = [f.result() for f in futures]

        assert all(status == "success" for status in results)


class TestErrorMessages:
    """Test error message quality and clarity."""

    def test_syntax_error_message_informative(self) -> None:
        """Test that syntax error messages are informative."""
        result = execute_code("def foo(", timeout=5.0)
        assert result["status"] == "syntax_error"
        assert result["error"] is not None
        assert len(result["error"]) > 0

    def test_runtime_error_includes_type(self) -> None:
        """Test that runtime errors include exception type."""
        result = execute_code("raise KeyError('missing')", timeout=5.0)
        assert result["status"] == "runtime_error"
        assert "KeyError" in result["error"]

    def test_timeout_error_clear(self) -> None:
        """Test that timeout errors have clear messages."""
        result = execute_code("while True: pass", timeout=0.5)
        assert result["status"] == "timeout"
        assert "timeout" in result["error"].lower() or "timed out" in result["error"].lower()


@pytest.mark.slow
class TestPerformanceCharacteristics:
    """Test performance characteristics and overheads."""

    def test_spawn_overhead_acceptable(self) -> None:
        """Test that spawn overhead is within acceptable bounds."""
        start = time.time()
        result = execute_code("x = 1", timeout=10.0)
        elapsed = time.time() - start

        assert result["status"] == "success"
        # Spawn overhead should be < 15s (typically ~7s + some margin)
        assert elapsed < 15.0

    def test_multiple_executions_amortize_cost(self) -> None:
        """Test that multiple executions complete in reasonable time."""
        start = time.time()

        for i in range(3):
            result = execute_code(f"x = {i}", timeout=5.0)
            assert result["status"] == "success"

        elapsed = time.time() - start
        # 3 executions with ~7s spawn each = ~21s, add margin
        assert elapsed < 45.0
