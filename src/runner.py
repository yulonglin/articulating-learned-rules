"""
Experiment runner with CLI arguments and JSONL output.

Features:
- Argparse for all parameters
- JSONL output format
- Proper logging with timestamps and parameters
- Support for batch processing
- Reproducibility tracking (git commit, timestamp, all params)
"""

import argparse
import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

from src.api_caller import (
    CacheMode,
    Message,
    create_caller,
)
from src.model_registry import DEFAULT_TEST_MODEL


@dataclass
class ExperimentConfig:
    """Configuration for experiment run."""
    # Experiment metadata
    experiment_name: str
    output_dir: Path
    model: str

    # API configuration
    temperature: float
    max_tokens: int
    cache_mode: CacheMode
    cache_dir: Path
    max_concurrent: int

    # Experiment parameters
    random_seed: int
    max_samples: Optional[int]
    batch_size: int

    # Logging
    log_level: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cache_mode": self.cache_mode.value,
            "cache_dir": str(self.cache_dir),
            "max_concurrent": self.max_concurrent,
            "random_seed": self.random_seed,
            "max_samples": self.max_samples,
            "batch_size": self.batch_size,
            "log_level": self.log_level,
        }


class ExperimentMetadata(BaseModel):
    """Metadata for experiment run."""
    experiment_name: str
    timestamp: str
    git_commit: Optional[str]
    config: dict[str, Any]


class ExperimentResult(BaseModel):
    """Single experiment result in JSONL format."""
    sample_id: int
    input: str
    output: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    metadata: dict[str, Any] = {}


class ExperimentRunner:
    """Runner for LLM experiments with JSONL output."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Create API caller
        self.caller = create_caller(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            cache_mode=config.cache_mode,
            cache_dir=config.cache_dir,
            max_concurrent=config.max_concurrent,
        )

        # Initialize output file
        self.results_file = self.output_dir / "results.jsonl"
        self.metadata_file = self.output_dir / "metadata.json"

        # Log experiment start
        self._log_experiment_start()

    def _setup_logging(self) -> None:
        """Setup logging to file and console."""
        log_file = self.output_dir / "experiment.log"

        # Create logger
        self.logger = logging.getLogger(self.config.experiment_name)
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))

        # Remove existing handlers
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _log_experiment_start(self) -> None:
        """Log experiment metadata and configuration."""
        git_commit = self._get_git_commit()

        metadata = ExperimentMetadata(
            experiment_name=self.config.experiment_name,
            timestamp=datetime.now().isoformat(),
            git_commit=git_commit,
            config=self.config.to_dict(),
        )

        # Save metadata
        with self.metadata_file.open("w") as f:
            f.write(metadata.model_dump_json(indent=2))

        # Log to console
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Model: {self.config.model}")
        self.logger.info(f"Random seed: {self.config.random_seed}")
        if git_commit:
            self.logger.info(f"Git commit: {git_commit}")
        self.logger.info(f"Configuration: {json.dumps(self.config.to_dict(), indent=2)}")

    def _append_result(self, result: ExperimentResult) -> None:
        """Append result to JSONL file."""
        with self.results_file.open("a") as f:
            f.write(result.model_dump_json() + "\n")

    async def run_single(
        self,
        sample_id: int,
        input_text: str,
        messages_fn: Callable[[str], list[Message]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> ExperimentResult:
        """
        Run single sample through API.

        Args:
            sample_id: Unique identifier for sample
            input_text: Input text to process
            messages_fn: Function to convert input to messages
            metadata: Additional metadata to store

        Returns:
            ExperimentResult with response
        """
        messages = messages_fn(input_text)
        response = await self.caller.call(messages)

        result = ExperimentResult(
            sample_id=sample_id,
            input=input_text,
            output=response.content,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cached=response.cached,
            metadata=metadata or {},
        )

        # Append to JSONL immediately
        self._append_result(result)

        return result

    async def run_batch(
        self,
        inputs: list[str],
        messages_fn: Callable[[str], list[Message]],
        metadata_fn: Optional[Callable[[int, str], dict[str, Any]]] = None,
        start_id: int = 0,
    ) -> list[ExperimentResult]:
        """
        Run batch of samples through API.

        Args:
            inputs: List of input texts
            messages_fn: Function to convert input to messages
            metadata_fn: Function to generate metadata for each sample
            start_id: Starting sample ID

        Returns:
            List of ExperimentResult objects
        """
        self.logger.info(f"Running batch of {len(inputs)} samples")

        results = []
        for i, input_text in enumerate(inputs):
            sample_id = start_id + i
            metadata = metadata_fn(sample_id, input_text) if metadata_fn else None

            result = await self.run_single(
                sample_id=sample_id,
                input_text=input_text,
                messages_fn=messages_fn,
                metadata=metadata,
            )
            results.append(result)

        self.logger.info(f"Completed batch of {len(inputs)} samples")
        return results

    async def run_experiment(
        self,
        inputs: list[str],
        messages_fn: Callable[[str], list[Message]],
        metadata_fn: Optional[Callable[[int, str], dict[str, Any]]] = None,
    ) -> list[ExperimentResult]:
        """
        Run full experiment with batching and checkpointing.

        Args:
            inputs: List of input texts
            messages_fn: Function to convert input to messages
            metadata_fn: Function to generate metadata for each sample

        Returns:
            List of all ExperimentResult objects
        """
        # Apply max_samples limit
        if self.config.max_samples is not None:
            inputs = inputs[: self.config.max_samples]

        self.logger.info(f"Running experiment on {len(inputs)} samples")

        # Process in batches
        all_results = []
        batch_size = self.config.batch_size

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_results = await self.run_batch(
                inputs=batch_inputs,
                messages_fn=messages_fn,
                metadata_fn=metadata_fn,
                start_id=i,
            )
            all_results.extend(batch_results)

            # Log progress
            self.logger.info(
                f"Progress: {min(i + batch_size, len(inputs))}/{len(inputs)} samples"
            )

        self.logger.info("Experiment completed")
        self.logger.info(f"Results saved to: {self.results_file}")

        return all_results


def create_experiment_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Create ExperimentConfig from argparse arguments."""
    return ExperimentConfig(
        experiment_name=args.experiment_name,
        output_dir=Path(args.output_dir),
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache_mode=CacheMode(args.cache_mode),
        cache_dir=Path(args.cache_dir),
        max_concurrent=args.max_concurrent,
        random_seed=args.random_seed,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        log_level=args.log_level,
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with all experiment parameters."""
    parser = argparse.ArgumentParser(
        description="Run LLM experiment with JSONL output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment metadata
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name of experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TEST_MODEL,
        help="Model name (e.g., gpt-4.1-nano, claude-haiku-4.5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens",
    )

    # Cache configuration
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="short",
        choices=["none", "short", "persistent"],
        help="Cache mode: none, short (15min), persistent",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="Directory for cache files",
    )

    # Rate limiting
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API requests",
    )

    # Experiment parameters
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser


# Example usage
async def example_experiment():
    """Example experiment demonstrating usage."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args([
        "--experiment-name", "example_test",
        "--output-dir", "experiments/20251026_example_test",
        "--model", "gpt-4o-mini",
        "--max-samples", "5",
    ])

    # Create config
    config = create_experiment_config_from_args(args)

    # Create runner
    runner = ExperimentRunner(config)

    # Example inputs
    inputs = [
        "the cat sat on the mat",
        "THE DOG RAN FAST",
        "hello world",
        "GOODBYE WORLD",
        "testing lowercase",
    ]

    # Define message function
    def messages_fn(input_text: str) -> list[Message]:
        return [
            Message(
                role="user",
                content=f"Is this text all lowercase? Respond with just 'yes' or 'no'.\n\nText: {input_text}",
            )
        ]

    # Define metadata function
    def metadata_fn(sample_id: int, input_text: str) -> dict[str, Any]:
        return {
            "is_lowercase": input_text.islower(),
            "length": len(input_text),
        }

    # Run experiment
    results = await runner.run_experiment(
        inputs=inputs,
        messages_fn=messages_fn,
        metadata_fn=metadata_fn,
    )

    print(f"\nCompleted {len(results)} samples")
    print(f"Results saved to: {runner.results_file}")


if __name__ == "__main__":
    asyncio.run(example_experiment())
