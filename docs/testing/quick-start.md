# Testing Quick Start Guide

## 🚀 Running Tests

### Prerequisites
```bash
# Install test dependencies
uv pip install pytest pytest-asyncio boto3 moto[server]
```

### Run All Tests
```bash
# Fast unit tests only
uv run pytest tests/unit -v

# Integration tests
uv run pytest tests/test_integration_local.py -v

# Everything
uv run pytest tests/ -v
```

### Run Specific Tests
```bash
# Single test class
uv run pytest tests/test_integration_local.py::TestMinerValidatorIntegration -v

# Single test method
uv run pytest tests/test_integration_local.py::TestMinerValidatorIntegration::test_single_miner_startup -v

# Tests matching pattern
uv run pytest tests/ -k "miner" -v
```

### Debugging Tests
```bash
# Show print statements
uv run pytest tests/ -s

# Drop into debugger on failure
uv run pytest tests/ --pdb

# Verbose logging
uv run pytest tests/ -v --log-cli-level=DEBUG
```

## 📝 Writing Tests

### Basic Test Structure
```python
import pytest
from grail.mining.engine import MiningEngine

@pytest.mark.integration
class TestMiningBehavior:
    """Test mining engine behavior."""
    
    async def test_basic_mining(self, miner_service):
        """Test that miner generates valid outputs."""
        # Given
        problem = create_test_problem()
        
        # When
        result = await miner_service.process(problem)
        
        # Then
        assert result.is_valid
        assert len(result.rollouts) > 0
```

### Using Fixtures
```python
@pytest.mark.integration
async def test_with_fixtures(
    miner_service,      # Starts a miner
    validator_service,  # Starts a validator
    s3_client,         # S3 client for verification
    service_manager    # Access to service logs
):
    """Test using multiple fixtures."""
    # Wait for services
    await asyncio.sleep(5)
    
    # Check logs
    logs = service_manager.get_logs("miner")
    assert "Started successfully" in logs
```

### Custom Fixtures
```python
@pytest.fixture
async def custom_miner(service_manager, test_environment):
    """Start a miner with custom configuration."""
    env = os.environ.copy()
    env.update({
        "GRAIL_MODEL_NAME": "custom-model",
        "GRAIL_MAX_NEW_TOKENS": "200"
    })
    
    cmd = [sys.executable, "-m", "grail", "mine"]
    return service_manager.start_service("custom_miner", cmd, env)
```

## 🎯 Test Patterns

### Wait for Async Operations
```python
async def wait_for_condition(condition_fn, timeout=30):
    """Wait for a condition to become true."""
    start = time.time()
    while time.time() - start < timeout:
        if condition_fn():
            return True
        await asyncio.sleep(1)
    return False

# Usage
success = await wait_for_condition(
    lambda: len(s3_client.list_objects()["Contents"]) > 0
)
assert success, "No files uploaded to S3"
```

### Test Data Helpers
```python
def create_test_problem(difficulty="easy"):
    """Create a test SAT problem."""
    if difficulty == "easy":
        return {
            "num_variables": 3,
            "clauses": [[1, 2], [-1, 3], [2, -3]]
        }
    # ... more difficulties
```

### Parametrized Tests
```python
@pytest.mark.parametrize("num_miners,expected_windows", [
    (1, 1),
    (2, 2),
    (3, 3),
])
async def test_multiple_miners(num_miners, expected_windows, service_manager):
    """Test with different numbers of miners."""
    # Start miners
    miners = []
    for i in range(num_miners):
        miner = await start_test_miner(service_manager, i)
        miners.append(miner)
    
    # Verify windows
    windows = await wait_for_windows(expected_windows)
    assert len(windows) == expected_windows
```

## 🐛 Common Issues

### "Service failed to start"
```bash
# Check service logs
cat /tmp/grail_test_*/logs/miner.log

# Common causes:
# - Model not downloaded
# - Port already in use
# - Missing dependencies
```

### "Timeout waiting for files"
```python
# Increase timeout in wait functions
files = await wait_for_s3_files(
    s3_client,
    prefix="grail/windows/",
    timeout=120  # Increase from default 60
)
```

### "Flaky tests"
```python
# Add retries for flaky operations
@pytest.mark.flaky(reruns=3, reruns_delay=2)
async def test_potentially_flaky():
    # Test that might fail due to timing
    pass
```

## 📊 Test Organization

```
tests/
├── conftest.py           # Shared fixtures
├── unit/                 # Fast, isolated tests
│   ├── test_protocol.py
│   └── test_models.py
├── integration/          # Component integration
│   ├── test_mining.py
│   └── test_validation.py
├── system/              # Full system tests
│   └── test_e2e.py
└── utils/               # Test helpers
    ├── factories.py
    └── assertions.py
```

## 🔧 Configuration

### pytest.ini Settings
```ini
[pytest]
# Run async tests automatically
asyncio_mode = auto

# Default markers
markers =
    integration: Integration tests
    slow: Long-running tests
    unit: Unit tests

# Timeout for tests
timeout = 300
```

### Environment Variables
```bash
# Speed up tests
export GRAIL_WINDOW_LENGTH=3
export GRAIL_MAX_NEW_TOKENS=50
export GRAIL_MODEL_NAME="Qwen/Qwen2-0.5B-Instruct"

# Disable external services
export WANDB_MODE=disabled
export GRAIL_MONITORING_BACKEND=null
```

## 📈 Measuring Test Quality

### Coverage
```bash
# Run with coverage
uv run pytest tests/ --cov=grail --cov-report=html

# View report
open htmlcov/index.html
```

### Performance
```bash
# Profile slow tests
uv run pytest tests/ --durations=10
```

### Parallel Execution
```bash
# Install pytest-xdist
uv pip install pytest-xdist

# Run tests in parallel
uv run pytest tests/ -n auto
```

## 🎓 Learning Resources

- [Testing Strategy Document](./testing_strategy.md)
- [ADR-001: Testing Approach](./adr-001-testing-approach.md)
- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)
