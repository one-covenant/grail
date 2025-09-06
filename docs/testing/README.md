# Grail Testing Documentation

Welcome to the Grail testing documentation. This folder contains comprehensive guides and references for testing the Grail Bittensor subnet.

## 📚 Documentation Structure

### [Quick Start Guide](./quick-start.md)
**Start here!** A concise guide to get you running tests immediately.
- Running tests
- Writing your first test
- Common patterns
- Troubleshooting

### [Testing Strategy](./testing_strategy.md)
Comprehensive overview of our testing approach:
- Testing challenges in Bittensor
- Multiple testing approaches analyzed
- Recommended three-tier strategy
- Implementation guidelines
- Best practices

### [ADR-001: Testing Approach](./adr-001-testing-approach.md)
Architectural Decision Record explaining why we chose pytest-based integration testing:
- Context and constraints
- Decision rationale
- Consequences and trade-offs
- Implementation details

### [Tier 3 Testing Guide](./tier3-guide.md)
Guide for running full integration tests with real miners and validators:
- Multiple model testing
- WandB integration
- Debug mode with `-vv` output
- Real-time monitoring

## 🎯 Quick Reference

### Run Tests
```bash
# All tests
uv run pytest tests/ -v

# Only integration tests
uv run pytest tests/test_integration_local.py -v

# Specific test
uv run pytest tests/test_integration_local.py::TestMinerValidatorIntegration::test_single_miner_startup -v
```

### Test Structure
```
tests/
├── conftest.py                    # Fixtures and configuration
├── test_integration_local.py      # Main integration tests
├── test_integration_simple.py     # Simple verification tests
└── README.md                      # Test-specific documentation
```

### Key Concepts

1. **Fixtures**: Reusable test components (services, mocks, environments)
2. **Service Manager**: Handles process lifecycle for miners/validators
3. **Mock S3**: Uses moto instead of real S3/MinIO
4. **Tiered Testing**: Unit → Integration → System tests

## 🔍 Navigation Guide

- **New to testing?** → Start with [Quick Start Guide](./quick-start.md)
- **Want to understand the approach?** → Read [ADR-001](./adr-001-testing-approach.md)
- **Need comprehensive details?** → See [Testing Strategy](./testing_strategy.md)
- **Writing tests?** → Check examples in `/tests/` directory

## 🤝 Contributing

When adding new tests:
1. Follow the patterns in existing tests
2. Use appropriate fixtures
3. Add documentation for complex scenarios
4. Ensure tests are deterministic
5. Keep tests fast and focused

## 📊 Testing Philosophy

Our testing approach follows these principles:

1. **Fast Feedback**: Most tests should run in seconds
2. **Realistic Behavior**: Test actual code paths, not just mocks
3. **Maintainability**: Tests should be as clean as production code
4. **Documentation**: Tests serve as living documentation
5. **Determinism**: Tests should not be flaky

## 🚀 Future Improvements

Planned enhancements to our testing infrastructure:

- [ ] Performance benchmarking suite
- [ ] Chaos testing for network failures
- [ ] Load testing for scalability
- [ ] Visual test reporting dashboard
- [ ] Integration with CI/CD metrics

## 📞 Getting Help

- Check the [Quick Start Guide](./quick-start.md) for common issues
- Review test examples in `/tests/` directory
- Read pytest output carefully - it often has the solution
- Use `--pdb` flag to debug failing tests

---

*Last updated: September 2025*
