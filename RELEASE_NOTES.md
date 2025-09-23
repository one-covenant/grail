# GRAIL v0.0.15 Release Notes

## 🚨 Critical Fix

- **Validator weight submission restored**: Fixed accidentally commented out `set_weights` call in validation service that prevented validators from submitting weights to the network

---

**⚠️ URGENT UPDATE REQUIRED**: All validators must update immediately to restore network weight submission functionality.

```bash
git pull && uv sync --all-extras
# Restart your validator services
```

---

## Previous Release (v0.0.14)

### 🔧 Critical Fixes

- **Group ordering mismatch resolved**: Fixed type mismatch in GRPO group ordering between validators and miners using content-based deterministic sorting
- **Miner rewards simplified**: Rewards now based solely on unique rollouts submitted - success/failure no longer impacts scores

### 🛠️ Improvements

- Enhanced logging with miner context and UID tracking
- Reduced false positives in token distribution validation
- Fixed Drand generation hash configuration
- Increased SAT problem difficulty