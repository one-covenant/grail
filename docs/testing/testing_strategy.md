# Grail Testing Strategy

## Document Information
- **Type**: Testing Strategy & Architecture Document
- **Status**: Draft
- **Created**: September 2025
- **Purpose**: Define comprehensive testing approaches for the Grail Bittensor subnet

## Executive Summary

This document outlines various testing strategies for the Grail Bittensor subnet, considering the unique challenges of testing decentralized AI systems. It provides a recommended tiered approach that balances thoroughness with development velocity.

## Table of Contents
1. [Testing Challenges in Bittensor](#testing-challenges-in-bittensor)
2. [Testing Approaches](#testing-approaches)
3. [Recommended Strategy](#recommended-strategy)
4. [Implementation Guidelines](#implementation-guidelines)
5. [Best Practices](#best-practices)

## Testing Challenges in Bittensor

Testing Bittensor subnets presents unique challenges:

- **Decentralized Infrastructure**: Tests must account for blockchain interactions
- **Resource Intensity**: Mining and validation require significant computational resources
- **Network Effects**: Behavior depends on multiple interacting nodes
- **External Dependencies**: Chain state, storage systems, and model inference
- **Non-Determinism**: Network latency, chain finality, and model outputs

## Testing Approaches

### 1. Full Process Integration

**Description**: Run actual miner and validator processes with real components.

```python
# Real processes, mock external services
@pytest.fixture
def full_integration_env():
    miners = [start_miner_process(i) for i in range(2)]
    validator = start_validator_process()
    return {"miners": miners, "validator": validator}
```

**Pros**:
- Tests real code paths
- Catches integration issues
- Validates resource usage

**Cons**:
- Slow execution (minutes per test)
- Resource intensive (GPU/CPU)
- Complex setup and teardown

**Best For**: End-to-end validation, release testing, performance benchmarking

### 2. Bittensor Mock Layer

**Description**: Mock the entire Bittensor SDK to simulate chain interactions.

```python
class MockSubtensor:
    def __init__(self):
        self.neurons = {}
        self.weights = {}
    
    def metagraph(self, netuid):
        return MockMetagraph()
    
    def set_weights(self, wallet, netuid, weights):
        self.weights[wallet.hotkey] = weights
        return True
```

**Pros**:
- Fast execution (seconds)
- Deterministic results
- No chain dependency

**Cons**:
- May miss real chain interaction issues
- Requires maintaining mock accuracy

**Best For**: Unit tests, CI/CD pipelines, rapid development

### 3. Local Chain Simulation

**Description**: Run a local Bittensor test chain for realistic testing.

```python
class LocalBittensorChain:
    def __init__(self):
        self.process = subprocess.Popen([
            "substrate-node", 
            "--dev", 
            "--instant-seal",
            "--tmp"
        ])
        self.wait_for_ready()
    
    def fund_wallet(self, wallet, amount=1000):
        # Pre-fund test wallets
        pass
```

**Pros**:
- Real chain behavior
- Full control over chain state
- Instant block production

**Cons**:
- Complex setup
- Requires substrate node
- Additional maintenance burden

**Best For**: Protocol-level testing, chain interaction validation

### 4. Hybrid Mock/Real Components

**Description**: Use real Grail logic with mocked Bittensor components.

```python
@pytest.fixture
def hybrid_environment(monkeypatch):
    # Mock only Bittensor-specific parts
    monkeypatch.setattr("bittensor.subtensor", MockSubtensor)
    monkeypatch.setattr("bittensor.wallet", MockWallet)
    
    # Real grail components
    from grail.mining.engine import MiningEngine
    from grail.validation.engine import ValidationEngine
    
    return {
        "mining_engine": MiningEngine(),
        "validation_engine": ValidationEngine(),
        "mock_chain": MockSubtensor()
    }
```

**Pros**:
- Tests core logic without chain dependencies
- Faster than full integration
- Good balance of realism and speed

**Cons**:
- May miss chain-specific edge cases
- Requires careful mock design

**Best For**: Logic validation, component testing, fast feedback

### 5. Scenario-Based Testing

**Description**: Define test scenarios as declarative configurations.

```yaml
scenarios:
  happy_path:
    miners: 2
    validator: 1
    rounds: 3
    expected:
      windows_generated: 6
      valid_rollouts: ">0"
      
  miner_failure:
    miners: 3
    validator: 1
    fail_miner_at_round: 2
    expected:
      recovery: true
      windows_after_failure: 4
```

**Pros**:
- Declarative and readable
- Easy to add new scenarios
- Good for QA and regression testing

**Cons**:
- Less flexible for complex tests
- May require scenario runner framework

**Best For**: Regression testing, QA validation, behavior documentation

### 6. Record/Replay Testing

**Description**: Record real network interactions for replay testing.

```python
class NetworkRecorder:
    def __init__(self):
        self.recordings = []
    
    def record_interaction(self, interaction_type, data):
        self.recordings.append({
            "timestamp": time.time(),
            "type": interaction_type,
            "data": data
        })
    
    def save_recording(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.recordings, f)
```

**Pros**:
- Tests against real production data
- Helps debug production issues
- Can validate backwards compatibility

**Cons**:
- Recordings can become stale
- Large storage requirements
- Privacy considerations

**Best For**: Debugging production issues, regression validation

### 7. Container-Based Testing

**Description**: Use Docker Compose with test-specific configurations.

```yaml
# docker-compose.test.yml
version: '3.8'
services:
  test-runner:
    build: .
    command: pytest /app/tests/integration
    environment:
      - GRAIL_TEST_MODE=true
    depends_on:
      - test-chain
      - test-storage
      
  test-chain:
    image: bittensor/local-chain:latest
    command: --instant-seal --test-mode
```

**Pros**:
- Isolated environment
- Reproducible across systems
- Close to production setup

**Cons**:
- Requires Docker
- Not suitable for Docker-in-Docker
- Slower than in-process tests

**Best For**: CI/CD with proper Docker support, integration testing

### 8. Lightweight Service Simulation

**Description**: Simulate services without full process overhead.

```python
class LightweightMiner:
    def __init__(self, wallet, model_name="mock"):
        self.wallet = wallet
        self.model_name = model_name
    
    async def generate_rollout(self, problem):
        # Use cached responses or tiny model
        if self.model_name == "mock":
            return self._mock_rollout(problem)
        else:
            return self._tiny_model_rollout(problem)
```

**Pros**:
- Very fast execution
- Low resource usage
- Good for testing protocol logic

**Cons**:
- Not testing real model behavior
- May miss performance issues

**Best For**: Protocol logic testing, rapid iteration

## Recommended Strategy

Based on Bittensor ecosystem best practices, we recommend a **three-tier testing pyramid**:

### Tier 1: Fast Unit/Integration Tests (Base of Pyramid)
- **Frequency**: Every commit
- **Duration**: < 30 seconds total
- **Approach**: Bittensor Mock Layer (#2) + Lightweight Simulation (#8)

```
tests/
├── unit/
│   ├── test_grail_protocol.py
│   ├── test_rollout_generation.py
│   └── test_verification.py
└── fast_integration/
    ├── test_miner_validator_mock.py
    └── test_reward_calculation.py
```

### Tier 2: Component Integration Tests (Middle)
- **Frequency**: Pull request / merge
- **Duration**: < 5 minutes
- **Approach**: Hybrid Mock/Real (#4) + Scenario-Based (#5)

```
tests/
└── integration/
    ├── test_mining_engine.py
    ├── test_validation_flow.py
    └── scenarios/
        ├── happy_path.yaml
        └── edge_cases.yaml
```

### Tier 3: Full System Tests (Top)
- **Frequency**: Nightly / release
- **Duration**: < 30 minutes
- **Approach**: Full Process Integration (#1) or Container-Based (#7)

```
tests/
└── system/
    ├── test_full_network.py
    ├── test_performance.py
    └── test_chain_integration.py
```

## Implementation Guidelines

### 1. Test Data Management

```python
# tests/fixtures/test_data.py
TEST_WALLETS = {
    "miner_1": {
        "coldkey": "test_miner_1",
        "hotkey": "test_hot_1"
    },
    "validator": {
        "coldkey": "test_validator",
        "hotkey": "test_val_hot"
    }
}

TEST_PROBLEMS = {
    "simple_sat": {
        "num_variables": 3,
        "num_clauses": 5
    }
}
```

### 2. Mock Factories

```python
# tests/mocks/factories.py
class BittensorMockFactory:
    @staticmethod
    def create_mock_neuron(uid, stake=100.0, **kwargs):
        return MockNeuron(
            uid=uid,
            stake=stake,
            **kwargs
        )
    
    @staticmethod
    def create_mock_metagraph(n_neurons=256):
        neurons = [
            BittensorMockFactory.create_mock_neuron(i)
            for i in range(n_neurons)
        ]
        return MockMetagraph(neurons)
```

### 3. Test Utilities

```python
# tests/utils/assertions.py
def assert_valid_window_file(window_data):
    """Assert that a window file has valid structure."""
    assert "wallet" in window_data
    assert "window_start" in window_data
    assert "inferences" in window_data
    assert isinstance(window_data["inferences"], list)

def assert_rollout_verified(rollout, grail_instance):
    """Assert that a rollout passes GRAIL verification."""
    result = grail_instance.verify_output(
        rollout["problem"],
        rollout["output"],
        rollout["proof"]
    )
    assert result.is_valid, f"Rollout verification failed: {result.reason}"
```

## Best Practices

### 1. Bittensor-Specific Testing

- **Wallet Management**
  - Use deterministic test wallets (fixed mnemonics)
  - Test wallet creation, loading, and error scenarios
  - Mock signing operations for speed

- **Chain State**
  - Test with various metagraph configurations
  - Simulate stake and weight changes
  - Test chain disconnection/reconnection

- **Incentive Mechanisms**
  - Verify reward calculations are correct
  - Test edge cases (zero weights, all equal weights)
  - Validate that incentives align with desired behavior

### 2. Performance Testing

```python
@pytest.mark.benchmark
def test_rollout_generation_performance(benchmark):
    engine = MiningEngine(model="tiny")
    problem = generate_test_problem()
    
    result = benchmark(engine.generate_rollout, problem)
    
    assert result.time < 1.0  # Should complete in < 1 second
    assert result.memory < 100  # MB
```

### 3. Deterministic Testing

```python
@pytest.fixture
def deterministic_env(monkeypatch):
    """Ensure deterministic behavior in tests."""
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr("time.time", lambda: 1234567890)
    monkeypatch.setattr("random.random", lambda: 0.5)
    
    import torch
    torch.manual_seed(42)
    
    import numpy as np
    np.random.seed(42)
```

### 4. Resource Management

```python
@pytest.fixture(scope="session")
def shared_model():
    """Share expensive resources across tests."""
    model = load_test_model()
    yield model
    model.cleanup()

@pytest.mark.usefixtures("shared_model")
class TestModelBehavior:
    """Tests that share the same model instance."""
    pass
```

### 5. Error Injection

```python
class ChaosMockSubtensor(MockSubtensor):
    """Mock that randomly fails to simulate network issues."""
    
    def __init__(self, failure_rate=0.1):
        super().__init__()
        self.failure_rate = failure_rate
    
    def set_weights(self, *args, **kwargs):
        if random.random() < self.failure_rate:
            raise Exception("Network timeout")
        return super().set_weights(*args, **kwargs)
```

## Continuous Improvement

1. **Metrics to Track**
   - Test execution time
   - Test flakiness rate
   - Code coverage
   - Time to first failure

2. **Regular Reviews**
   - Monthly review of test failures
   - Quarterly assessment of test strategy
   - Continuous refactoring of test code

3. **Documentation**
   - Keep test scenarios documented
   - Maintain troubleshooting guide
   - Document known issues and workarounds

## Conclusion

This testing strategy provides a comprehensive approach to testing Bittensor subnets while balancing thoroughness with development velocity. The tiered approach ensures fast feedback for developers while maintaining confidence in the system's behavior through deeper integration tests.
