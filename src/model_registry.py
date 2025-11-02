"""
Model Registry - Central configuration for all LLM model names.

Provides constants for model identifiers and metadata to ensure consistency
across the codebase and simplify model updates.
"""


class GPTModels:
    """OpenAI GPT model identifiers."""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_1_NANO = "gpt-4.1-nano-2025-04-14"
    GPT_5_NANO = "gpt-5-nano-2025-08-07"


class ClaudeModels:
    """Anthropic Claude model identifiers."""

    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5-20251001"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"


# Default models for different use cases
DEFAULT_JUDGE_MODEL = GPTModels.GPT_4_1_NANO
DEFAULT_TEST_MODEL = GPTModels.GPT_4_1_NANO
DEFAULT_MULTI_MODEL_LIST = [GPTModels.GPT_4_1_NANO, ClaudeModels.CLAUDE_HAIKU_4_5]

# Display names for visualizations
MODEL_DISPLAY_NAMES = {
    GPTModels.GPT_4O_MINI: "GPT-4o-mini",
    GPTModels.GPT_4O: "GPT-4o",
    GPTModels.GPT_4_TURBO: "GPT-4-turbo",
    GPTModels.GPT_4_1_NANO: "GPT-4.1-nano",
    GPTModels.GPT_5_NANO: "GPT-5-nano",
    ClaudeModels.CLAUDE_3_5_HAIKU: "Claude-3.5-Haiku",
    ClaudeModels.CLAUDE_HAIKU_4_5: "Claude-Haiku-4.5",
    ClaudeModels.CLAUDE_3_5_SONNET: "Claude-3.5-Sonnet",
    ClaudeModels.CLAUDE_3_OPUS: "Claude-3-Opus",
}


def is_gpt_model(model: str) -> bool:
    """Check if model is an OpenAI GPT model.

    Args:
        model: Model identifier string

    Returns:
        True if model is a GPT model, False otherwise
    """
    return "gpt" in model.lower()


def is_claude_model(model: str) -> bool:
    """Check if model is an Anthropic Claude model.

    Args:
        model: Model identifier string

    Returns:
        True if model is a Claude model, False otherwise
    """
    return "claude" in model.lower()


def requires_new_api(model: str) -> bool:
    """Check if model requires new OpenAI API parameters.

    Models like gpt-4.1+ and gpt-5+ require additional API parameters
    compared to older models.

    Args:
        model: Model identifier string

    Returns:
        True if model requires new API parameters, False otherwise
    """
    return "gpt-4.1" in model or "gpt-5" in model


def get_display_name(model: str) -> str:
    """Get human-readable display name for a model.

    Args:
        model: Model identifier string

    Returns:
        Display name if found in registry, otherwise returns the model string as-is
    """
    return MODEL_DISPLAY_NAMES.get(model, model)
