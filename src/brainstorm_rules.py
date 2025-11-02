"""
LLM-based rule brainstorming for classification tasks.

Generates diverse, articulable classification rules using GPT-4.1-nano and Claude Haiku 4.5.
Each rule includes natural language articulation, category, examples, and difficulty rating.

Prompt Strategies:
1. Category-specific prompts (syntactic, pattern, semantic, statistical)
2. Complexity-specific prompts (simple, moderate, complex)
3. Input-type specific prompts (word-level, sentence-level, mixed)

Output Format:
- JSONL file with one rule per line
- Each rule includes: id, name, articulation, category, examples, difficulty, metadata
"""

import argparse
import asyncio
import json
import logging
import sys
# Removed unused imports - now using pure Pydantic
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from tqdm import tqdm

from src.api_caller import APICallerBase, CacheMode, Message, create_caller
from src.model_registry import DEFAULT_MULTI_MODEL_LIST, GPTModels, ClaudeModels
from src.utils import create_experiment_timestamp


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RuleExample(BaseModel):
    """Single example for a classification rule."""
    input: str
    label: bool


class ClassificationRule(BaseModel):
    """Structured classification rule with metadata."""
    rule_id: str
    rule_name: str
    articulation: str
    category: Literal["syntactic", "pattern", "semantic", "statistical"]
    expected_difficulty: Literal["easy", "moderate", "hard"]
    source_model: str
    timestamp: str
    prompt_strategy: str  # Which prompt strategy generated this rule
    examples: list[RuleExample] = Field(default_factory=list, max_length=5)


class BrainstormConfig(BaseModel):
    """Configuration for rule brainstorming."""
    output_path: Path
    num_rules_target: int = 100
    models: list[str] = DEFAULT_MULTI_MODEL_LIST
    cache_mode: CacheMode = CacheMode.SHORT
    temperature: float = 0.7  # Higher for creativity
    max_tokens: int = 4096
    batch_size: int = 10  # Rules per LLM call
    max_concurrent: int = 5  # Lower for higher temperature calls


# Prompt templates for diverse rule generation
PROMPT_TEMPLATES = {
    "syntactic_simple": """Generate {n} simple syntactic classification rules for text inputs.

Requirements for each rule:
- Must be based on syntactic features (capitalization, punctuation, word count, character patterns, etc.)
- Must be deterministic and easy for humans to verify
- Must be articulable in 1-2 clear sentences
- Should be learnable from 3-5 examples
- Difficulty: easy to moderate

For EACH rule, provide:
1. rule_name: Short identifier (e.g., "all_caps", "starts_with_vowel")
2. articulation: Clear 1-2 sentence description of when input is labeled True
3. category: Must be "syntactic"
4. examples: Exactly 3-5 examples with input string and true/false label
5. expected_difficulty: "easy" or "moderate"

Output as JSON array of {n} rules. Example format:
[
  {{
    "rule_name": "all_lowercase",
    "articulation": "The input is labeled True if all alphabetic characters are lowercase.",
    "category": "syntactic",
    "examples": [
      {{"input": "hello world", "label": true}},
      {{"input": "Hello World", "label": false}},
      {{"input": "hello123", "label": true}}
    ],
    "expected_difficulty": "easy"
  }}
]

Generate {n} DIVERSE syntactic rules now:""",

    "pattern_moderate": """Generate {n} pattern-based classification rules for text inputs.

Requirements for each rule:
- Must be based on patterns (repeating characters, specific sequences, regular expressions, etc.)
- Must be deterministic and verifiable
- Must be articulable in 1-2 clear sentences
- Should be learnable from 3-5 examples
- Difficulty: moderate

For EACH rule, provide:
1. rule_name: Short identifier
2. articulation: Clear description of the pattern
3. category: Must be "pattern"
4. examples: Exactly 3-5 examples with input string and true/false label
5. expected_difficulty: "moderate"

Output as JSON array of {n} rules with same format as above.

Generate {n} DIVERSE pattern rules now:""",

    "semantic_moderate": """Generate {n} semantic classification rules for text inputs.

Requirements for each rule:
- Must be based on meaning/semantics (topic, sentiment, intent, etc.)
- Should still be relatively deterministic and clear
- Must be articulable in 1-2 clear sentences
- Should be learnable from 3-5 examples
- Difficulty: moderate to hard

For EACH rule, provide:
1. rule_name: Short identifier
2. articulation: Clear description of the semantic criterion
3. category: Must be "semantic"
4. examples: Exactly 3-5 examples with input string and true/false label
5. expected_difficulty: "moderate" or "hard"

Output as JSON array of {n} rules with same format as above.

Generate {n} DIVERSE semantic rules now:""",

    "statistical_hard": """Generate {n} statistical classification rules for text inputs.

Requirements for each rule:
- Must be based on statistical properties (character frequency, word length distribution, etc.)
- Must be deterministic and calculable
- Must be articulable in 1-2 clear sentences
- Should be learnable from 3-5 examples
- Difficulty: moderate to hard

For EACH rule, provide:
1. rule_name: Short identifier
2. articulation: Clear description of the statistical criterion
3. category: Must be "statistical"
4. examples: Exactly 3-5 examples with input string and true/false label
5. expected_difficulty: "moderate" or "hard"

Output as JSON array of {n} rules with same format as above.

Generate {n} DIVERSE statistical rules now:""",

    "word_level": """Generate {n} classification rules that operate on individual words (not sentences).

Requirements for each rule:
- Input is a single word
- Can be syntactic, pattern, or semantic
- Must be deterministic and clear
- Must be articulable in 1-2 clear sentences
- Difficulty: any

For EACH rule, provide:
1. rule_name: Short identifier
2. articulation: Clear description
3. category: "syntactic", "pattern", "semantic", or "statistical"
4. examples: Exactly 3-5 examples with SINGLE WORD inputs and true/false label
5. expected_difficulty: "easy", "moderate", or "hard"

Output as JSON array of {n} rules with same format as above.

Generate {n} DIVERSE word-level rules now:""",

    "sentence_level": """Generate {n} classification rules that operate on full sentences.

Requirements for each rule:
- Input is a complete sentence
- Can be syntactic, pattern, or semantic
- Must be deterministic when possible
- Must be articulable in 1-2 clear sentences
- Difficulty: any

For EACH rule, provide:
1. rule_name: Short identifier
2. articulation: Clear description
3. category: "syntactic", "pattern", "semantic", or "statistical"
4. examples: Exactly 3-5 examples with SENTENCE inputs and true/false label
5. expected_difficulty: "easy", "moderate", or "hard"

Output as JSON array of {n} rules with same format as above.

Generate {n} DIVERSE sentence-level rules now:""",

    "creative_diverse": """Generate {n} creative and diverse classification rules for text inputs.

Requirements:
- Be creative and think of unusual but clear rules
- Mix syntactic, pattern, semantic, and statistical approaches
- All rules must still be deterministic or nearly deterministic
- Must be articulable in 1-2 clear sentences
- Cover various difficulty levels

For EACH rule, provide:
1. rule_name: Short identifier
2. articulation: Clear description
3. category: "syntactic", "pattern", "semantic", or "statistical"
4. examples: Exactly 3-5 examples with inputs and true/false labels
5. expected_difficulty: "easy", "moderate", or "hard"

Output as JSON array of {n} rules with same format as above.

Generate {n} DIVERSE creative rules now:"""
}


class RuleBrainstormer:
    """Generate diverse classification rules using LLMs."""

    def __init__(self, config: BrainstormConfig):
        self.config = config
        self.callers: dict[str, APICallerBase] = {}
        self.generated_rules: list[ClassificationRule] = []
        self.rule_names_seen: set[str] = set()

        # Initialize API callers
        for model in config.models:
            self.callers[model] = create_caller(
                model=model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                cache_mode=config.cache_mode,
                max_concurrent=config.max_concurrent,
            )

        logger.info(f"Initialized brainstormer with models: {config.models}")

    def _create_prompt(self, strategy: str, batch_size: int) -> Message:
        """Create prompt for a specific strategy."""
        template = PROMPT_TEMPLATES[strategy]
        user_content = template.format(n=batch_size)
        return Message(role="user", content=user_content)

    def _parse_rule_response(
        self,
        response_content: str,
        source_model: str,
        prompt_strategy: str,
    ) -> list[ClassificationRule]:
        """Parse LLM response into ClassificationRule objects."""
        rules: list[ClassificationRule] = []

        # Try to parse as JSON array
        try:
            # Extract JSON from markdown code blocks if present
            content = response_content.strip()
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            data = json.loads(content)
            if not isinstance(data, list):
                logger.warning(f"Expected JSON array, got {type(data)}")
                return rules

            timestamp = datetime.now().isoformat()

            for i, rule_dict in enumerate(data):
                # Generate unique rule ID
                rule_name = rule_dict.get("rule_name", f"unnamed_{i}")
                rule_id = f"{rule_name}_{source_model.split('-')[0]}_{len(self.generated_rules) + len(rules):03d}"

                # Convert examples to RuleExample objects
                examples = []
                for ex in rule_dict.get("examples", []):
                    examples.append(RuleExample(
                        input=ex["input"],
                        label=ex["label"]
                    ))

                # Create ClassificationRule
                rule = ClassificationRule(
                    rule_id=rule_id,
                    rule_name=rule_name,
                    articulation=rule_dict["articulation"],
                    category=rule_dict["category"],
                    examples=examples,
                    expected_difficulty=rule_dict["expected_difficulty"],
                    source_model=source_model,
                    timestamp=timestamp,
                    prompt_strategy=prompt_strategy,
                )

                rules.append(rule)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {response_content[:500]}")
        except Exception as e:
            logger.error(f"Error parsing rule: {e}")
            logger.debug(f"Response content: {response_content[:500]}")

        return rules

    def _deduplicate_rule(self, rule: ClassificationRule) -> bool:
        """
        Check if rule is duplicate based on rule_name.

        Returns:
            True if rule is unique, False if duplicate
        """
        if rule.rule_name in self.rule_names_seen:
            logger.debug(f"Skipping duplicate rule: {rule.rule_name}")
            return False

        self.rule_names_seen.add(rule.rule_name)
        return True

    async def generate_rules_batch(
        self,
        model: str,
        strategy: str,
        batch_size: int,
    ) -> list[ClassificationRule]:
        """Generate a batch of rules using a specific model and strategy."""
        caller = self.callers[model]
        prompt = self._create_prompt(strategy, batch_size)

        logger.info(f"Generating {batch_size} rules with {model} using strategy: {strategy}")

        response = await caller.call([prompt])

        # Parse response
        rules = self._parse_rule_response(
            response.content,
            source_model=model,
            prompt_strategy=strategy,
        )

        # Deduplicate
        unique_rules = [r for r in rules if self._deduplicate_rule(r)]

        logger.info(f"Generated {len(unique_rules)} unique rules from {model} ({strategy})")

        return unique_rules

    async def generate_all_rules(self) -> list[ClassificationRule]:
        """Generate rules using all models and strategies until target is met."""
        strategies = list(PROMPT_TEMPLATES.keys())

        # Calculate how many batches we need
        # Distribute across models and strategies
        # Add buffer for deduplication
        total_batches_needed = (self.config.num_rules_target + self.config.batch_size - 1) // self.config.batch_size
        max_batches = total_batches_needed * 2  # *2 buffer for deduplication

        batch_idx = 0

        with tqdm(total=self.config.num_rules_target, desc="Generating rules", disable=not sys.stdout.isatty()) as pbar:
            while len(self.generated_rules) < self.config.num_rules_target and batch_idx < max_batches:
                # Create a batch of tasks (up to 10 at a time)
                tasks = []
                for _ in range(min(10, max_batches - batch_idx)):
                    if len(self.generated_rules) >= self.config.num_rules_target:
                        break

                    model = self.config.models[batch_idx % len(self.config.models)]
                    strategy = strategies[batch_idx % len(strategies)]

                    # Create fresh coroutine for each task
                    task = self.generate_rules_batch(
                        model=model,
                        strategy=strategy,
                        batch_size=self.config.batch_size,
                    )
                    tasks.append(task)
                    batch_idx += 1

                # Run this batch of tasks
                if tasks:
                    prev_count = len(self.generated_rules)
                    batch_results = await asyncio.gather(*tasks)
                    for rules in batch_results:
                        self.generated_rules.extend(rules)

                    # Update progress bar by the number of new rules
                    new_count = len(self.generated_rules)
                    pbar.update(new_count - prev_count)

                    logger.info(f"Progress: {len(self.generated_rules)}/{self.config.num_rules_target} rules generated")

        logger.info(f"Completed: {len(self.generated_rules)} total rules generated")

        # Trim to target if we exceeded
        return self.generated_rules[:self.config.num_rules_target]

    def save_rules(self, rules: list[ClassificationRule]) -> None:
        """Save rules to JSONL file."""
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.config.output_path.open("w") as f:
            for rule in rules:
                f.write(rule.model_dump_json() + "\n")

        logger.info(f"Saved {len(rules)} rules to {self.config.output_path}")


async def main(args: argparse.Namespace) -> None:
    """Main execution function."""
    # Parse models
    if args.models == "both":
        models = DEFAULT_MULTI_MODEL_LIST
    elif args.models == "gpt":
        models = [GPTModels.GPT_4_1_NANO]
    elif args.models == "claude":
        models = [ClaudeModels.CLAUDE_HAIKU_4_5]
    else:
        models = args.models.split(",")

    # Create output path with timestamp if not specified
    if args.output is None:
        timestamp = create_experiment_timestamp()
        output_path = Path("out") / f"brainstormed_rules_{timestamp}.jsonl"
    else:
        output_path = Path(args.output)

    # Create config
    config = BrainstormConfig(
        num_rules_target=args.num_rules_target,
        models=models,
        output_path=output_path,
        cache_mode=CacheMode(args.cache_mode),
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    logger.info(f"Starting rule brainstorming with config:")
    logger.info(f"  Target rules: {config.num_rules_target}")
    logger.info(f"  Models: {config.models}")
    logger.info(f"  Output: {config.output_path}")
    logger.info(f"  Temperature: {config.temperature}")
    logger.info(f"  Batch size: {config.batch_size}")

    # Generate rules
    brainstormer = RuleBrainstormer(config)
    rules = await brainstormer.generate_all_rules()

    # Save results
    brainstormer.save_rules(rules)

    # Print summary statistics
    logger.info("\n=== Generation Summary ===")
    logger.info(f"Total rules: {len(rules)}")

    # Category distribution
    category_counts = {}
    for rule in rules:
        category_counts[rule.category] = category_counts.get(rule.category, 0) + 1
    logger.info(f"Categories: {category_counts}")

    # Difficulty distribution
    difficulty_counts = {}
    for rule in rules:
        difficulty_counts[rule.expected_difficulty] = difficulty_counts.get(rule.expected_difficulty, 0) + 1
    logger.info(f"Difficulties: {difficulty_counts}")

    # Model distribution
    model_counts = {}
    for rule in rules:
        model_name = rule.source_model.split("-")[0]  # gpt or claude
        model_counts[model_name] = model_counts.get(model_name, 0) + 1
    logger.info(f"Models: {model_counts}")

    logger.info(f"\nOutput saved to: {config.output_path.absolute()}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Brainstorm classification rules using LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-rules-target",
        type=int,
        default=100,
        help="Target number of rules to generate",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="both",
        choices=["both", "gpt", "claude"],
        help="Which models to use for generation",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (default: tmp/brainstormed_rules_TIMESTAMP.jsonl)",
    )

    parser.add_argument(
        "--cache-mode",
        type=str,
        default="short",
        choices=["none", "short", "persistent"],
        help="Cache mode for API calls",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for LLM generation (higher = more creative)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of rules to request per LLM call",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))


# USAGE EXAMPLES:
#
# Generate 100 rules using both GPT-4o-mini and Claude 3.5 Haiku:
#   uv run python -m src.brainstorm_rules --num-rules-target 100
#
# Generate 50 rules using only GPT-4o-mini:
#   uv run python -m src.brainstorm_rules --num-rules-target 50 --models gpt
#
# Generate rules with higher temperature for more creativity:
#   uv run python -m src.brainstorm_rules --temperature 0.9
#
# Specify custom output path:
#   uv run python -m src.brainstorm_rules --output experiments/rules_v1.jsonl
#
# Use persistent caching (good for iterative development):
#   uv run python -m src.brainstorm_rules --cache-mode persistent
#
# Request more rules per batch (faster but may reduce diversity):
#   uv run python -m src.brainstorm_rules --batch-size 20
