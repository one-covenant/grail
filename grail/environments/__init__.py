"""GRAIL Environments - Step-only RL environments for various tasks.

Exports:
- Core API: MultiTurnEnv, SingleTurnEnv, types
- SAT: problem generator, parser, reward, env (all from sat_env)
- GSM8K: dataset-backed env for grade school math
- MATH: Hendrycks MATH benchmark env with multi-strategy validation
- PythonCode: MBPP/HumanEval dataset-backed env for code generation
- Loop: AgentEnvLoop, GRPORollout
- Factory: create_env, create_env_factory (preferred for instantiation)
- Legacy: Parser, RewardVector
"""

from .affinetes import (  # noqa: F401
    AffineLogicEnv,
    AffineTraceEnv,
    LogicTaskSource,
    TraceTaskSource,
)
from .base import Parser, RewardVector  # noqa: F401
from .core import (  # noqa: F401
    ChatMessage,
    MultiTurnEnv,
    Observation,
    RewardBreakdown,
    Rubric,
    SingleTurnEnv,
    TaskSource,
)
from .dataset_base import MathDatasetEnv  # noqa: F401
from .factory import (  # noqa: F401
    clear_task_source_cache,
    create_env,
    create_env_factory,
    get_or_create_task_source,
)
from .gpu_kernel import (  # noqa: F401
    EvalResult,
    KernelBenchTaskSource,
    KernelEvalBackend,
    TritonKernelEnv,
    TritonKernelParser,
    TritonKernelRubric,
    UnifiedKernelTaskSource,
    create_backend,
    create_triton_kernel_reward_vector,
    get_global_backend,
    set_global_backend,
    validate_gpu_config,
)
from .gsm8k_env import GSM8KEnv  # noqa: F401
from .loop import AgentEnvLoop, GRPORollout  # noqa: F401
from .math_hendrycks_env import MATHEnv  # noqa: F401
from .providers import (  # noqa: F401
    GSM8KTaskSource,
    HumanEvalTaskSource,
    MATHTaskSource,
    MBPPTaskSource,
    SATTaskSource,
)
from .python_code_env import PythonCodeEnv  # noqa: F401
from .rubric import RewardVectorRubric  # noqa: F401
from .sat_env import (  # noqa: F401
    SATEnv,
    SATParser,
    SATProblem,
    create_sat_prompt,
    create_sat_reward_vector,
    generate_sat_problem,
    sat_correctness_reward,
)


def get_sat_reward_bounds() -> tuple[float, float]:
    """Return SAT reward bounds or permissive defaults on failure.

    Used by validators to configure reward bounds for SAT validation.

    Returns:
        Tuple of (low, high) reward bounds as floats.
        Returns (-inf, inf) if bounds cannot be determined.
    """
    try:
        sat_rv = create_sat_reward_vector()
        low, high = sat_rv.reward_bounds()
        return float(low), float(high)
    except Exception:
        return float("-inf"), float("inf")


__all__ = [
    # Base classes (legacy)
    "Parser",
    "RewardVector",
    # Core step-only API
    "ChatMessage",
    "Observation",
    "RewardBreakdown",
    "TaskSource",
    "Rubric",
    "MultiTurnEnv",
    "SingleTurnEnv",
    "MathDatasetEnv",
    # Providers and rubrics
    "SATTaskSource",
    "GSM8KTaskSource",
    "MATHTaskSource",
    "MBPPTaskSource",
    "HumanEvalTaskSource",
    "RewardVectorRubric",
    # Affinetes adapters
    "AffineTraceEnv",
    "AffineLogicEnv",
    "TraceTaskSource",
    "LogicTaskSource",
    # Environments
    "SATEnv",
    "GSM8KEnv",
    "MATHEnv",
    "PythonCodeEnv",
    "TritonKernelEnv",
    # GPU Kernel providers
    "KernelBenchTaskSource",
    "UnifiedKernelTaskSource",
    "TritonKernelParser",
    "TritonKernelRubric",
    "create_triton_kernel_reward_vector",
    # GPU Kernel eval backends
    "EvalResult",
    "KernelEvalBackend",
    "create_backend",
    "get_global_backend",
    "set_global_backend",
    "validate_gpu_config",
    # Factory functions (preferred)
    "create_env",
    "create_env_factory",
    "get_or_create_task_source",
    "clear_task_source_cache",
    # SAT public API (validators use these)
    "SATProblem",
    "SATParser",
    "generate_sat_problem",
    "sat_correctness_reward",
    "create_sat_reward_vector",
    "create_sat_prompt",
    # Loop and rollouts
    "AgentEnvLoop",
    "GRPORollout",
    # Helper functions
    "get_sat_reward_bounds",
]
