# ADR-001: Testing Approach for Grail Integration Tests

## Status
Proposed

## Context

The Grail Bittensor subnet requires comprehensive testing to ensure reliability of the decentralized AI training system. The original approach used Docker Compose for integration testing, but we encountered Docker-in-Docker limitations in the development environment.

### Constraints
1. Development environment runs inside a Docker container
2. Docker-in-Docker is not available due to security restrictions
3. Need to test miner-validator interactions
4. Must validate GRAIL protocol implementation
5. Tests should be runnable in CI/CD pipelines

### Requirements
- Fast feedback for developers
- Comprehensive integration testing
- Resource efficiency
- Maintainability
- Compatibility with various environments

## Decision

We will implement a **pytest-based integration testing framework** that runs services as local processes rather than Docker containers. This approach uses:

1. **Process-based service management** via Python subprocess
2. **Moto for S3 mocking** instead of MinIO containers
3. **Pytest fixtures** for lifecycle management
4. **Tiered testing strategy** as documented in testing_strategy.md

## Consequences

### Positive
- ✅ No Docker-in-Docker requirement
- ✅ Faster test execution (no container overhead)
- ✅ Better debugging capabilities (direct process access)
- ✅ Easier to run in various CI/CD environments
- ✅ More Pythonic and maintainable
- ✅ Better integration with Python testing ecosystem

### Negative
- ❌ Less production-like than container-based tests
- ❌ May miss container-specific issues
- ❌ Requires careful process cleanup
- ❌ Different from production deployment

### Neutral
- 🔄 Requires different skills (pytest vs Docker Compose)
- 🔄 Changes how developers run integration tests
- 🔄 Different resource management approach

## Implementation Details

### Service Management
```python
# Instead of Docker Compose
# docker-compose -f docker-compose.integration.yml up

# We use pytest fixtures
@pytest.fixture
async def miner_service(service_manager):
    return service_manager.start_service("miner", cmd, env)
```

### Storage Mocking
```python
# Instead of MinIO container
# services:
#   s3:
#     image: minio/minio:latest

# We use moto
@pytest.fixture
def mock_s3_server():
    # Start moto server for S3 mocking
```

### Test Execution
```bash
# Instead of
docker-compose -f docker-compose.integration.yml up
docker-compose -f docker-compose.integration.yml down

# We use
uv run pytest tests/test_integration_local.py
# Cleanup is automatic via fixtures
```

## Alternatives Considered

### 1. Fix Docker-in-Docker
- Required privileged containers or special Docker configuration
- Not portable across environments
- Security concerns

### 2. Use Testcontainers
- Still requires Docker access
- Same Docker-in-Docker issues
- Additional dependency

### 3. Mock Everything
- Too far from real behavior
- Wouldn't catch integration issues
- Low confidence in tests

### 4. Manual Testing Only
- Not scalable
- Not reproducible
- No CI/CD integration

## References

- [Testing Strategy Document](./testing_strategy.md)
- [Pytest Best Practices](https://docs.pytest.org/en/stable/explanation/practices.html)
- [Moto - Mock AWS Services](https://github.com/getmoto/moto)
- [Bittensor Testing Guidelines](https://github.com/opentensor/bittensor)

## Notes

This decision can be revisited if:
1. Docker-in-Docker becomes available
2. We need more production-like testing
3. We encounter process-based testing limitations
4. Better alternatives emerge

For teams with proper Docker environments, the original docker-compose.integration.yml approach remains valid and can be used alongside these pytest-based tests.
