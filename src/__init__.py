"""
Articulating Learned Rules - Research Infrastructure

Core modules for LLM rule articulation experiments.
"""

from src.api_caller import (
    APICallerBase,
    APIConfig,
    APIResponse,
    AnthropicCaller,
    CacheMode,
    DEFAULT_TEMPERATURE,
    Message,
    OpenAICaller,
    create_caller,
)
from src.model_registry import (
    ClaudeModels,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MULTI_MODEL_LIST,
    DEFAULT_TEST_MODEL,
    GPTModels,
    MODEL_DISPLAY_NAMES,
    get_display_name,
    is_claude_model,
    is_gpt_model,
    requires_new_api,
)
_runner_exports: list[str] = []
try:  # pragma: no cover - optional dependency setup
    from src.runner import (
        ExperimentConfig,
        ExperimentMetadata,
        ExperimentResult,
        ExperimentRunner,
        create_argument_parser,
        create_experiment_config_from_args,
    )

    _runner_exports = [
        "ExperimentConfig",
        "ExperimentMetadata",
        "ExperimentResult",
        "ExperimentRunner",
        "create_argument_parser",
        "create_experiment_config_from_args",
    ]
except ImportError:
    _runner_exports = []

__all__ = [
    # API Caller
    "APICallerBase",
    "APIConfig",
    "APIResponse",
    "AnthropicCaller",
    "CacheMode",
    "DEFAULT_TEMPERATURE",
    "Message",
    "OpenAICaller",
    "create_caller",
    # Model Registry
    "ClaudeModels",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_MULTI_MODEL_LIST",
    "DEFAULT_TEST_MODEL",
    "GPTModels",
    "MODEL_DISPLAY_NAMES",
    "get_display_name",
    "is_claude_model",
    "is_gpt_model",
    "requires_new_api",
    # Runner
    *_runner_exports,
]
