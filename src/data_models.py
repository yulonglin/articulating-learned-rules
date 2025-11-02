"""
Shared data models for articulating-learned-rules experiments.

This module provides common Pydantic models used across different
evaluation scripts to ensure consistency and reduce duplication.
"""

from typing import Any, Optional

from pydantic import BaseModel


class Rule(BaseModel):
    """
    Classification rule definition used across evaluation scripts.

    This model represents a curated classification rule with all metadata
    needed for learnability, articulation, and faithfulness testing.

    Attributes:
        rule_id: Unique identifier for the rule
        rule_name: Human-readable name
        articulation: Natural language description of the rule
        category: Rule category (syntactic, pattern, semantic, statistical)
        examples: List of example input/label pairs
        expected_difficulty: Predicted difficulty level (easy, moderate, hard)
        learnability: Optional model-specific learnability metadata added by
                     enrich_rules_with_learnability.py. Maps model names to
                     their min_few_shot_required and best_accuracy values.
    """

    rule_id: str
    rule_name: str
    articulation: str
    category: str
    examples: list[dict[str, Any]]
    expected_difficulty: str
    learnability: Optional[dict[str, dict[str, Any]]] = None


class DatasetSample(BaseModel):
    """Single sample from a generated dataset."""

    input: str
    label: bool
    rule_id: str
    metadata: dict[str, Any] = {}
