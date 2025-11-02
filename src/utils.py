"""
Utility functions for experiment setup and data processing.
"""

import json
import random
from pathlib import Path
from typing import Any, Iterator

import yaml
from pydantic import BaseModel


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    # Note: If using numpy or torch, also set their seeds here


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """
    Load JSONL file into list of dictionaries.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries from JSONL
    """
    results = []
    with file_path.open("r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def iter_jsonl(file_path: Path) -> Iterator[dict[str, Any]]:
    """
    Iterate over JSONL file without loading entire file into memory.

    Args:
        file_path: Path to JSONL file

    Yields:
        Dictionary for each line
    """
    with file_path.open("r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_jsonl(data: list[dict[str, Any]], file_path: Path) -> None:
    """
    Save list of dictionaries to JSONL file.

    Args:
        data: List of dictionaries to save
        file_path: Output file path
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_yaml(file_path: Path) -> dict[str, Any]:
    """
    Load YAML file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary from YAML
    """
    with file_path.open("r") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], file_path: Path) -> None:
    """
    Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def update_registry(
    registry_path: Path,
    experiment_name: str,
    metadata: dict[str, Any],
) -> None:
    """
    Update experiment registry with new experiment.

    Args:
        registry_path: Path to registry YAML file
        experiment_name: Name of experiment
        metadata: Experiment metadata to add
    """
    # Load existing registry
    if registry_path.exists():
        registry = load_yaml(registry_path)
    else:
        registry = {"experiments": []}

    # Ensure experiments key exists
    if "experiments" not in registry:
        registry["experiments"] = []

    # Add or update experiment
    existing_idx = None
    for i, exp in enumerate(registry["experiments"]):
        if exp.get("name") == experiment_name:
            existing_idx = i
            break

    if existing_idx is not None:
        registry["experiments"][existing_idx] = metadata
    else:
        registry["experiments"].append(metadata)

    # Save updated registry
    save_yaml(registry, registry_path)


def compute_accuracy(
    predictions: list[str], labels: list[str], normalize_fn: callable = str.lower
) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: List of predicted labels
        labels: List of true labels
        normalize_fn: Function to normalize strings before comparison

    Returns:
        Accuracy as float between 0 and 1
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(
        1
        for pred, label in zip(predictions, labels)
        if normalize_fn(pred) == normalize_fn(label)
    )
    return correct / len(predictions)


def extract_yes_no(response: str) -> str:
    """
    Extract yes/no answer from response text.

    Args:
        response: Model response text

    Returns:
        "yes" or "no" (normalized to lowercase)
    """
    response_lower = response.lower().strip()

    # Direct match
    if response_lower in ["yes", "no"]:
        return response_lower

    # Contains yes/no
    if "yes" in response_lower and "no" not in response_lower:
        return "yes"
    if "no" in response_lower and "yes" not in response_lower:
        return "no"

    # Default to first word
    first_word = response_lower.split()[0] if response_lower.split() else ""
    if first_word in ["yes", "no"]:
        return first_word

    # Unable to parse
    return response_lower


def create_experiment_timestamp() -> str:
    """
    Create timestamp string for experiment naming.

    Returns:
        Timestamp in YYYYMMDD_HHMMSS format
    """
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")
