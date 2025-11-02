"""
Dataset generation tool for classification rules.

Generates fixed, high-quality datasets for classification experiments with:
- Balanced positive/negative cases
- Edge case coverage
- Diversity in inputs
- Deterministic labeling

Supports both programmatic and LLM-based generation strategies.
"""

import argparse
import asyncio
import json
import random
import re
import statistics
import string
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

from src.api_caller import Message, create_caller, CacheMode
from src.model_registry import DEFAULT_TEST_MODEL
from src.utils import save_jsonl, save_yaml, set_random_seed


# ============================================================================
# Data Models
# ============================================================================


class Rule(BaseModel):
    """Classification rule definition."""
    rule_id: str
    rule_name: str  # Use rule_name from curated file
    articulation: str  # Use articulation as description
    category: str = "text"  # syntactic, semantic, pattern, statistical
    implementability: str = "programmatic"  # programmatic, llm_needed, complex

    # Derived properties
    @property
    def name(self) -> str:
        """Get rule name."""
        return self.rule_name

    @property
    def description(self) -> str:
        """Get rule description."""
        return self.articulation

    @property
    def programmatic(self) -> bool:
        """Check if can be generated programmatically."""
        return self.implementability == "programmatic"


class DatasetSample(BaseModel):
    """Single labeled sample in dataset."""
    input: str
    label: bool
    rule_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetMetadata(BaseModel):
    """Metadata for generated dataset."""
    rule_id: str
    rule_name: str
    rule_description: str
    num_samples: int
    num_positive: int
    num_negative: int
    generation_method: Literal["programmatic", "llm", "hybrid"]
    models_used: list[str] = Field(default_factory=list)
    random_seed: int
    timestamp: str
    quality_checks: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Rule Evaluation Functions
# ============================================================================

# Common stopwords for stopword_ratio rule
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'should', 'could', 'may', 'might', 'must', 'can', 'it', 'its',
    'that', 'this', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they'
}


# ===== SYNTACTIC RULES =====

def eval_all_caps(text: str) -> bool:
    """Check if all alphabetic characters are uppercase."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False  # Require at least one alphabetic character
    return all(c.isupper() for c in alpha_chars)


def eval_contains_digit(text: str) -> bool:
    """Check if text contains at least one digit."""
    return any(c.isdigit() for c in text)


def eval_ends_with_period(text: str) -> bool:
    """Check if text ends with a period."""
    if not text:
        return False
    return text[-1] == '.'


def eval_word_count_greater_than_five(text: str) -> bool:
    """Check if text has more than 5 words."""
    words = text.split()
    return len(words) > 5


def eval_contains_exclamation(text: str) -> bool:
    """Check if text contains at least one exclamation mark."""
    return '!' in text


def eval_contains_special_character(text: str) -> bool:
    """Check if text contains at least one special character."""
    special_chars = set('@#$%&*!?^~`|\\/<>[]{}()+=_-')
    return any(c in special_chars for c in text)


def eval_no_spaces(text: str) -> bool:
    """Check if text contains no spaces."""
    return ' ' not in text


def eval_multiple_sentences(text: str) -> bool:
    """Check if text contains more than one sentence (multiple periods)."""
    return text.count('.') > 1


def eval_word_count_between_3_and_7(text: str) -> bool:
    """Check if word count is between 3 and 7 inclusive."""
    words = text.split()
    return 3 <= len(words) <= 7


# ===== PATTERN RULES =====

def eval_starts_with_vowel(text: str) -> bool:
    """Check if text starts with a vowel (case-insensitive)."""
    if not text:
        return False
    return text[0].lower() in 'aeiou'


def eval_consecutive_repeated_chars(text: str) -> bool:
    """Check if text has 3+ identical consecutive characters."""
    for i in range(len(text) - 2):
        if text[i] == text[i+1] == text[i+2]:
            return True
    return False


def eval_contains_negation(text: str) -> bool:
    """Check if text contains negation words."""
    negation_words = {'not', 'no', 'never', "n't"}
    words = set(text.lower().split())
    # Also check for contractions with n't
    text_lower = text.lower()
    return bool(words & negation_words) or "n't" in text_lower


def eval_numeric_palindrome(text: str) -> bool:
    """Check if numeric string is palindrome with at least 3 digits."""
    # Extract only digits
    digits = ''.join(c for c in text if c.isdigit())
    if len(digits) < 3:
        return False
    return digits == digits[::-1]


def eval_repeated_substring(text: str) -> bool:
    """Check if string contains a substring (len >= 2) that repeats at least twice."""
    n = len(text)
    for length in range(2, n // 2 + 1):
        for i in range(n - length + 1):
            substring = text[i:i+length]
            # Check if this substring appears elsewhere
            rest = text[:i] + text[i+length:]
            if substring in rest:
                return True
    return False


def eval_symmetric_character_count(text: str) -> bool:
    """Check if first and second half have equal character counts."""
    mid = len(text) // 2
    first_half = text[:mid]
    second_half = text[mid:] if len(text) % 2 == 0 else text[mid+1:]
    return len(first_half) == len(second_half)


def eval_special_character_bookends(text: str) -> bool:
    """Check if text starts and ends with special char, alphanumeric in between."""
    if len(text) < 3:
        return False
    special_chars = set('@#$%&*!?^~`|\\/<>[]{}()+=_-')
    if text[0] not in special_chars or text[-1] not in special_chars:
        return False
    # Check middle has alphanumeric
    middle = text[1:-1]
    return any(c.isalnum() for c in middle)


def eval_is_anagram_of_list(text: str) -> bool:
    """Check if text is anagram of 'listen'."""
    target = 'listen'
    # Remove spaces and convert to lowercase
    cleaned = text.replace(' ', '').lower()
    return sorted(cleaned) == sorted(target)


def eval_rhyming_ends(text: str) -> bool:
    """Check if last words of two consecutive lines rhyme (simple heuristic)."""
    lines = text.split('\n')
    if len(lines) < 2:
        return False

    # Get last words of first two lines
    words1 = lines[0].strip().split()
    words2 = lines[1].strip().split()
    if not words1 or not words2:
        return False

    last1 = words1[-1].lower().rstrip('.,!?;:')
    last2 = words2[-1].lower().rstrip('.,!?;:')

    # Simple rhyme check: last 2-3 characters match
    if len(last1) >= 2 and len(last2) >= 2:
        return last1[-2:] == last2[-2:] or last1[-3:] == last2[-3:]
    return False


def eval_negation_presence(text: str) -> bool:
    """Check if text contains any negation words (same as contains_negation)."""
    return eval_contains_negation(text)


# ===== STATISTICAL RULES =====

def eval_avg_word_length(text: str) -> bool:
    """Classify as 'complex' if avg word length > 6."""
    words = text.split()
    if not words:
        return False
    avg_len = sum(len(w) for w in words) / len(words)
    return avg_len > 6


def eval_char_freq_vowel_ratio(text: str) -> bool:
    """Classify as 'vowel-heavy' if vowel ratio > 0.4."""
    if not text:
        return False
    vowels = sum(1 for c in text.lower() if c in 'aeiou')
    return vowels / len(text) > 0.4


def eval_longest_word_length(text: str) -> bool:
    """Classify as 'advanced' if longest word > 12 chars."""
    words = text.split()
    if not words:
        return False
    return max(len(w) for w in words) > 12


def eval_word_repetition_rate(text: str) -> bool:
    """Classify as 'repetitive' if any word appears > 3 times."""
    words = [w.lower() for w in text.split()]
    if not words:
        return False
    counts = Counter(words)
    return max(counts.values()) > 3


def eval_unique_word_count_ratio(text: str) -> bool:
    """Classify as 'original' if unique/total ratio > 0.5."""
    words = [w.lower() for w in text.split()]
    if not words:
        return False
    return len(set(words)) / len(words) > 0.5


def eval_stopword_ratio(text: str) -> bool:
    """Classify as 'stopword-heavy' if stopword ratio > 0.3."""
    words = [w.lower() for w in text.split()]
    if not words:
        return False
    stopword_count = sum(1 for w in words if w in STOPWORDS)
    return stopword_count / len(words) > 0.3


def eval_sentence_length_variance(text: str) -> bool:
    """Classify as 'varied' if variance of sentence lengths > 10 words."""
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        return False

    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return False

    variance = statistics.variance(lengths)
    return variance > 10


def eval_word_length_variance(text: str) -> bool:
    """Classify as TRUE if word length variance is between 1.5 and 3.0."""
    words = text.split()
    if len(words) < 2:
        return False

    lengths = [len(w) for w in words]
    variance = statistics.variance(lengths)
    return 1.5 <= variance <= 3.0


def eval_lengthy_sentences(text: str) -> bool:
    """Classify as long if > 20 words."""
    words = text.split()
    return len(words) > 20


# Map rule names to evaluation functions
EVAL_FUNCTIONS: dict[str, Callable[[str], bool]] = {
    # Syntactic
    "all_caps": eval_all_caps,
    "contains_digit": eval_contains_digit,
    "ends_with_period": eval_ends_with_period,
    "word_count_greater_than_five": eval_word_count_greater_than_five,
    "contains_exclamation": eval_contains_exclamation,
    "contains_special_character": eval_contains_special_character,
    "no_spaces": eval_no_spaces,
    "multiple_sentences": eval_multiple_sentences,
    "word_count_between_3_and_7": eval_word_count_between_3_and_7,
    # Pattern
    "starts_with_vowel": eval_starts_with_vowel,
    "consecutive_repeated_chars": eval_consecutive_repeated_chars,
    "contains_negation": eval_contains_negation,
    "numeric_palindrome": eval_numeric_palindrome,
    "repeated_substring": eval_repeated_substring,
    "symmetric_character_count": eval_symmetric_character_count,
    "special_character_bookends": eval_special_character_bookends,
    "is_anagram_of_list": eval_is_anagram_of_list,
    "RhymingEnds": eval_rhyming_ends,
    "NegationPresence": eval_negation_presence,
    # Statistical
    "avg_word_length": eval_avg_word_length,
    "char_freq_vowel_ratio": eval_char_freq_vowel_ratio,
    "longest_word_length": eval_longest_word_length,
    "word_repetition_rate": eval_word_repetition_rate,
    "unique_word_count_ratio": eval_unique_word_count_ratio,
    "stopword_ratio": eval_stopword_ratio,
    "sentence_length_variance": eval_sentence_length_variance,
    "word_length_variance": eval_word_length_variance,
    "LengthySentences": eval_lengthy_sentences,
}


# LLM-needed rules (no programmatic eval)
LLM_EVAL_RULES = {
    "topic_health",
    "sentiment_positive",
    "intent_purchase",
    "topic_technology",
    "intent_inform",
    "intent_request",
    "is_adjective",
}


# ============================================================================
# Input Generators
# ============================================================================


class InputGenerator(ABC):
    """Abstract base class for input generators."""

    def __init__(self, rule: Rule, random_seed: int = 42):
        self.rule = rule
        self.random_seed = random_seed
        random.seed(random_seed)

    @abstractmethod
    async def generate(self, num_samples: int, target_label: bool) -> list[str]:
        """Generate inputs with target label."""
        pass


class ProgrammaticGenerator(InputGenerator):
    """Generate inputs programmatically for simple rules."""

    def __init__(self, rule: Rule, random_seed: int = 42):
        super().__init__(rule, random_seed)

        # Common word lists for natural text generation
        self.lowercase_words = [
            "hello", "world", "the", "quick", "brown", "fox", "jumps",
            "over", "lazy", "dog", "cat", "sat", "mat", "testing",
            "example", "sample", "text", "data", "input", "output", "python",
            "code", "run", "test", "check", "value", "result", "function"
        ]
        self.uppercase_words = [
            "HELLO", "WORLD", "THE", "QUICK", "BROWN", "FOX", "JUMPS",
            "TESTING", "EXAMPLE", "SAMPLE", "DATA", "INPUT", "OUTPUT", "PYTHON"
        ]
        self.long_words = [
            "programming", "international", "understanding", "communication",
            "extraordinarily", "unconventional", "sophisticated", "implementation",
            "comprehensive", "antidisestablishmentarianism", "incomprehensibility"
        ]
        self.short_words = ["a", "an", "the", "is", "to", "in", "on", "at", "it"]
        self.vowel_words = ["apple", "orange", "elephant", "umbrella", "ice", "example", "input", "output", "under", "over"]
        self.consonant_words = ["the", "quick", "brown", "fox", "testing", "sample", "dog", "cat", "python", "data"]

    async def generate(self, num_samples: int, target_label: bool) -> list[str]:
        """Generate inputs using rule-specific strategies."""
        rule_name = self.rule.name

        # Dispatch to appropriate generator
        generators = {
            "all_caps": self._gen_all_caps,
            "contains_digit": self._gen_contains_digit,
            "ends_with_period": self._gen_ends_with_period,
            "word_count_greater_than_five": self._gen_word_count_greater_than_five,
            "contains_exclamation": self._gen_contains_exclamation,
            "contains_special_character": self._gen_contains_special_character,
            "no_spaces": self._gen_no_spaces,
            "multiple_sentences": self._gen_multiple_sentences,
            "word_count_between_3_and_7": self._gen_word_count_between_3_and_7,
            "starts_with_vowel": self._gen_starts_with_vowel,
            "consecutive_repeated_chars": self._gen_consecutive_repeated_chars,
            "contains_negation": self._gen_contains_negation,
            "numeric_palindrome": self._gen_numeric_palindrome,
            "repeated_substring": self._gen_repeated_substring,
            "symmetric_character_count": self._gen_symmetric_character_count,
            "special_character_bookends": self._gen_special_character_bookends,
            "is_anagram_of_list": self._gen_is_anagram_of_list,
            "RhymingEnds": self._gen_rhyming_ends,
            "NegationPresence": self._gen_contains_negation,  # Same as contains_negation
            "avg_word_length": self._gen_avg_word_length,
            "char_freq_vowel_ratio": self._gen_char_freq_vowel_ratio,
            "longest_word_length": self._gen_longest_word_length,
            "word_repetition_rate": self._gen_word_repetition_rate,
            "unique_word_count_ratio": self._gen_unique_word_count_ratio,
            "stopword_ratio": self._gen_stopword_ratio,
            "sentence_length_variance": self._gen_sentence_length_variance,
            "word_length_variance": self._gen_word_length_variance,
            "LengthySentences": self._gen_lengthy_sentences,
        }

        if rule_name not in generators:
            raise ValueError(f"No programmatic generator for rule: {rule_name}")

        return generators[rule_name](num_samples, target_label)

    # ===== SYNTACTIC GENERATORS =====

    def _gen_all_caps(self, n: int, target: bool) -> list[str]:
        """Generate for all_caps rule."""
        results = []
        for _ in range(n):
            num_words = random.randint(2, 5)
            if target:  # All uppercase
                words = random.choices(self.uppercase_words, k=num_words)
                # Add numbers sometimes (still valid)
                if random.random() < 0.3:
                    words.append(str(random.randint(100, 999)))
            else:  # Has lowercase
                # Mix of cases
                words = random.choices(self.lowercase_words, k=num_words)
            results.append(" ".join(words))
        return results

    def _gen_contains_digit(self, n: int, target: bool) -> list[str]:
        """Generate for contains_digit rule."""
        results = []
        for _ in range(n):
            num_words = random.randint(2, 5)
            words = random.choices(self.lowercase_words, k=num_words)
            text = " ".join(words)
            if target:
                insert_pos = random.randint(0, len(text))
                digit = str(random.randint(0, 9))
                text = text[:insert_pos] + digit + text[insert_pos:]
            results.append(text)
        return results

    def _gen_ends_with_period(self, n: int, target: bool) -> list[str]:
        """Generate for ends_with_period rule."""
        results = []
        for _ in range(n):
            num_words = random.randint(2, 5)
            words = random.choices(self.lowercase_words, k=num_words)
            text = " ".join(words)
            if target:
                text = text + "."
            elif random.random() < 0.5:  # Sometimes end with other punct
                text = text + random.choice("!?,:;")
            results.append(text)
        return results

    def _gen_word_count_greater_than_five(self, n: int, target: bool) -> list[str]:
        """Generate for word_count_greater_than_five rule."""
        results = []
        for _ in range(n):
            if target:
                num_words = random.randint(6, 10)
            else:
                num_words = random.randint(1, 5)
            words = random.choices(self.lowercase_words, k=num_words)
            results.append(" ".join(words))
        return results

    def _gen_contains_exclamation(self, n: int, target: bool) -> list[str]:
        """Generate for contains_exclamation rule."""
        results = []
        for _ in range(n):
            num_words = random.randint(2, 5)
            words = random.choices(self.lowercase_words, k=num_words)
            text = " ".join(words)
            if target:
                insert_pos = random.randint(len(text)//2, len(text))
                text = text[:insert_pos] + "!" + text[insert_pos:]
            results.append(text)
        return results

    def _gen_contains_special_character(self, n: int, target: bool) -> list[str]:
        """Generate for contains_special_character rule."""
        special_chars = ['@', '#', '$', '%', '&', '*', '!', '?']
        results = []
        for _ in range(n):
            num_words = random.randint(2, 5)
            words = random.choices(self.lowercase_words, k=num_words)
            text = " ".join(words)
            if target:
                char = random.choice(special_chars)
                insert_pos = random.randint(0, len(text))
                text = text[:insert_pos] + char + text[insert_pos:]
            results.append(text)
        return results

    def _gen_no_spaces(self, n: int, target: bool) -> list[str]:
        """Generate for no_spaces rule."""
        results = []
        for _ in range(n):
            if target:
                length = random.randint(5, 15)
                chars = random.choices(string.ascii_lowercase, k=length)
                results.append("".join(chars))
            else:
                num_words = random.randint(2, 4)
                words = random.choices(self.lowercase_words, k=num_words)
                results.append(" ".join(words))
        return results

    def _gen_multiple_sentences(self, n: int, target: bool) -> list[str]:
        """Generate for multiple_sentences rule."""
        results = []
        for _ in range(n):
            if target:  # More than one period
                num_sentences = random.randint(2, 4)
            else:  # 0 or 1 period
                num_sentences = random.randint(0, 1)

            sentences = []
            for _ in range(num_sentences):
                num_words = random.randint(2, 5)
                words = random.choices(self.lowercase_words, k=num_words)
                sentences.append(" ".join(words))

            if num_sentences > 0:
                text = ". ".join(sentences) + "."
            else:
                text = " ".join(random.choices(self.lowercase_words, k=random.randint(2, 5)))
            results.append(text)
        return results

    def _gen_word_count_between_3_and_7(self, n: int, target: bool) -> list[str]:
        """Generate for word_count_between_3_and_7 rule."""
        results = []
        for _ in range(n):
            if target:
                num_words = random.randint(3, 7)
            else:
                num_words = random.choice([1, 2, 8, 9, 10])
            words = random.choices(self.lowercase_words, k=num_words)
            results.append(" ".join(words))
        return results

    # ===== PATTERN GENERATORS =====

    def _gen_starts_with_vowel(self, n: int, target: bool) -> list[str]:
        """Generate for starts_with_vowel rule."""
        results = []
        for _ in range(n):
            if target:
                first_word = random.choice(self.vowel_words)
            else:
                first_word = random.choice(self.consonant_words)
            num_more = random.randint(1, 4)
            more_words = random.choices(self.lowercase_words, k=num_more)
            results.append(f"{first_word} {' '.join(more_words)}")
        return results

    def _gen_consecutive_repeated_chars(self, n: int, target: bool) -> list[str]:
        """Generate for consecutive_repeated_chars rule."""
        results = []
        for _ in range(n):
            base_word = random.choice(self.lowercase_words)
            if target:
                # Insert 3+ repeated chars
                char = random.choice(string.ascii_lowercase)
                repeat_count = random.randint(3, 5)
                insert_pos = random.randint(0, len(base_word))
                text = base_word[:insert_pos] + char * repeat_count + base_word[insert_pos:]
            else:
                text = base_word
            results.append(text)
        return results

    def _gen_contains_negation(self, n: int, target: bool) -> list[str]:
        """Generate for contains_negation rule."""
        negation_words = ["not", "no", "never", "don't", "won't", "can't"]
        results = []
        for _ in range(n):
            num_words = random.randint(3, 6)
            words = random.choices(self.lowercase_words, k=num_words)
            if target:
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, random.choice(negation_words))
            results.append(" ".join(words))
        return results

    def _gen_numeric_palindrome(self, n: int, target: bool) -> list[str]:
        """Generate for numeric_palindrome rule."""
        results = []
        for _ in range(n):
            if target:
                # Generate palindrome
                half_len = random.randint(2, 4)
                half = "".join(str(random.randint(0, 9)) for _ in range(half_len))
                palindrome = half + half[::-1]
            else:
                # Non-palindrome
                length = random.randint(3, 7)
                palindrome = "".join(str(random.randint(0, 9)) for _ in range(length))
                # Ensure it's not accidentally a palindrome
                if palindrome == palindrome[::-1]:
                    palindrome = palindrome[:-1] + str((int(palindrome[-1]) + 1) % 10)
            results.append(palindrome)
        return results

    def _gen_repeated_substring(self, n: int, target: bool) -> list[str]:
        """Generate for repeated_substring rule."""
        results = []
        for _ in range(n):
            if target:
                substring = random.choice(["la", "na", "ab", "co", "ing", "an"])
                base = random.choice(self.lowercase_words)
                text = substring + base + substring
            else:
                text = random.choice(self.lowercase_words)
            results.append(text)
        return results

    def _gen_symmetric_character_count(self, n: int, target: bool) -> list[str]:
        """Generate for symmetric_character_count rule."""
        results = []
        for _ in range(n):
            if target:
                # Even length string
                half_len = random.randint(3, 6)
                half1 = "".join(random.choices(string.ascii_lowercase, k=half_len))
                half2 = "".join(random.choices(string.ascii_lowercase, k=half_len))
                text = half1 + half2
            else:
                # Odd length
                length = random.randint(5, 11)
                if length % 2 == 0:
                    length += 1
                text = "".join(random.choices(string.ascii_lowercase, k=length))
            results.append(text)
        return results

    def _gen_special_character_bookends(self, n: int, target: bool) -> list[str]:
        """Generate for special_character_bookends rule."""
        special_chars = ['@', '#', '$', '%', '&', '*', '!', '?']
        results = []
        for _ in range(n):
            base = random.choice(self.lowercase_words)
            if target:
                start = random.choice(special_chars)
                end = random.choice(special_chars)
                text = f"{start}{base}{end}"
            else:
                text = base
            results.append(text)
        return results

    def _gen_is_anagram_of_list(self, n: int, target: bool) -> list[str]:
        """Generate for is_anagram_of_list rule."""
        anagrams = ["silent", "enlist", "tinsel", "listen", "inlets"]
        results = []
        for _ in range(n):
            if target:
                text = random.choice(anagrams)
            else:
                text = random.choice(self.lowercase_words)
            results.append(text)
        return results

    def _gen_rhyming_ends(self, n: int, target: bool) -> list[str]:
        """Generate for RhymingEnds rule."""
        rhyme_pairs = [
            ("cat", "mat"), ("night", "fright"), ("day", "play"),
            ("sun", "fun"), ("tree", "free"), ("log", "dog")
        ]
        results = []
        for _ in range(n):
            if target:
                pair = random.choice(rhyme_pairs)
                line1 = " ".join([random.choice(self.lowercase_words) for _ in range(random.randint(2, 4))] + [pair[0]])
                line2 = " ".join([random.choice(self.lowercase_words) for _ in range(random.randint(2, 4))] + [pair[1]])
                text = f"{line1}\n{line2}"
            else:
                line1 = " ".join(random.choices(self.lowercase_words, k=random.randint(3, 5)))
                line2 = " ".join(random.choices(self.lowercase_words, k=random.randint(3, 5)))
                text = f"{line1}\n{line2}"
            results.append(text)
        return results

    # ===== STATISTICAL GENERATORS =====

    def _gen_avg_word_length(self, n: int, target: bool) -> list[str]:
        """Generate for avg_word_length rule (> 6 chars)."""
        results = []
        for _ in range(n):
            if target:  # Long words
                num_words = random.randint(3, 6)
                words = random.choices(self.long_words, k=num_words)
            else:  # Short words
                num_words = random.randint(3, 6)
                words = random.choices(self.short_words + self.lowercase_words[:10], k=num_words)
            results.append(" ".join(words))
        return results

    def _gen_char_freq_vowel_ratio(self, n: int, target: bool) -> list[str]:
        """Generate for char_freq_vowel_ratio rule (> 0.4)."""
        results = []
        for _ in range(n):
            if target:  # High vowel ratio
                num_words = random.randint(3, 5)
                words = random.choices(self.vowel_words + ["aaa", "eee", "ooo"], k=num_words)
            else:  # Low vowel ratio
                consonant_heavy = ["rhythm", "crypts", "fly", "gym", "dry", "sky"]
                num_words = random.randint(3, 5)
                words = random.choices(consonant_heavy, k=num_words)
            results.append(" ".join(words))
        return results

    def _gen_longest_word_length(self, n: int, target: bool) -> list[str]:
        """Generate for longest_word_length rule (> 12 chars)."""
        results = []
        for _ in range(n):
            if target:
                # Include a very long word
                num_words = random.randint(2, 4)
                words = random.choices(self.lowercase_words, k=num_words)
                words.append(random.choice(self.long_words))
                random.shuffle(words)
            else:
                num_words = random.randint(3, 6)
                words = random.choices(self.lowercase_words, k=num_words)
            results.append(" ".join(words))
        return results

    def _gen_word_repetition_rate(self, n: int, target: bool) -> list[str]:
        """Generate for word_repetition_rate rule (word appears > 3 times)."""
        results = []
        for _ in range(n):
            if target:
                repeat_word = random.choice(self.lowercase_words)
                num_repeats = random.randint(4, 6)
                other_words = random.choices(self.lowercase_words, k=random.randint(2, 4))
                words = [repeat_word] * num_repeats + other_words
                random.shuffle(words)
            else:
                num_words = random.randint(5, 8)
                words = random.choices(self.lowercase_words, k=num_words)
            results.append(" ".join(words))
        return results

    def _gen_unique_word_count_ratio(self, n: int, target: bool) -> list[str]:
        """Generate for unique_word_count_ratio rule (> 0.5)."""
        results = []
        for _ in range(n):
            if target:  # High unique ratio
                num_words = random.randint(6, 10)
                words = random.sample(self.lowercase_words, min(num_words, len(self.lowercase_words)))
            else:  # Low unique ratio (lots of repetition)
                base_words = random.sample(self.lowercase_words, 3)
                words = base_words * 3  # Each word repeated 3 times
                random.shuffle(words)
            results.append(" ".join(words))
        return results

    def _gen_stopword_ratio(self, n: int, target: bool) -> list[str]:
        """Generate for stopword_ratio rule (> 0.3)."""
        stopwords_list = list(STOPWORDS)
        results = []
        for _ in range(n):
            if target:  # High stopword ratio
                num_stopwords = random.randint(5, 8)
                num_content = random.randint(2, 4)
                words = random.choices(stopwords_list, k=num_stopwords) + random.choices(self.lowercase_words, k=num_content)
                random.shuffle(words)
            else:  # Low stopword ratio
                num_words = random.randint(4, 7)
                words = random.choices(self.lowercase_words, k=num_words)
            results.append(" ".join(words))
        return results

    def _gen_sentence_length_variance(self, n: int, target: bool) -> list[str]:
        """Generate for sentence_length_variance rule (variance > 10)."""
        results = []
        for _ in range(n):
            if target:  # High variance - very different sentence lengths
                sent1 = " ".join(random.choices(self.lowercase_words, k=2))
                sent2 = " ".join(random.choices(self.lowercase_words, k=15))
                sent3 = " ".join(random.choices(self.lowercase_words, k=5))
                text = f"{sent1}. {sent2}. {sent3}."
            else:  # Low variance - similar lengths
                sent1 = " ".join(random.choices(self.lowercase_words, k=4))
                sent2 = " ".join(random.choices(self.lowercase_words, k=5))
                sent3 = " ".join(random.choices(self.lowercase_words, k=4))
                text = f"{sent1}. {sent2}. {sent3}."
            results.append(text)
        return results

    def _gen_word_length_variance(self, n: int, target: bool) -> list[str]:
        """Generate for word_length_variance rule (1.5 <= variance <= 3.0)."""
        results = []
        for _ in range(n):
            if target:  # Moderate variance
                words = random.choices(self.lowercase_words + self.long_words[:3], k=random.randint(5, 8))
            else:  # Very low or very high variance
                if random.random() < 0.5:
                    # Very low variance - all similar length
                    words = random.choices(["the", "cat", "sat", "mat", "fox"], k=6)
                else:
                    # Very high variance - extremes
                    words = ["a", "an"] + random.choices(self.long_words, k=3)
            results.append(" ".join(words))
        return results

    def _gen_lengthy_sentences(self, n: int, target: bool) -> list[str]:
        """Generate for LengthySentences rule (> 20 words)."""
        results = []
        for _ in range(n):
            if target:
                num_words = random.randint(21, 35)
            else:
                num_words = random.randint(3, 20)
            words = random.choices(self.lowercase_words, k=num_words)
            results.append(" ".join(words))
        return results


class LLMGenerator(InputGenerator):
    """Generate inputs using LLM for complex/semantic rules."""

    def __init__(
        self,
        rule: Rule,
        model: str = DEFAULT_TEST_MODEL,
        random_seed: int = 42,
        cache_mode: CacheMode = CacheMode.PERSISTENT
    ):
        super().__init__(rule, random_seed)
        self.model = model
        self.caller = create_caller(
            model=model,
            temperature=0.7,  # Higher temp for diversity
            cache_mode=cache_mode,
        )

    async def generate(self, num_samples: int, target_label: bool) -> list[str]:
        """Generate inputs using LLM."""
        label_str = "matches" if target_label else "does not match"

        prompt = f"""Generate {num_samples} diverse text examples that {label_str} this rule:

Rule: {self.rule.description}

Requirements:
- Each example should be a short phrase or sentence (5-30 words)
- Examples should be diverse and natural
- Avoid repetitive patterns
- Include edge cases when possible

Return ONLY a JSON array of strings, one example per element. No explanations.

Example format: ["example 1", "example 2", ...]"""

        messages = [Message(role="user", content=prompt)]
        response = await self.caller.call(messages)

        # Parse JSON response
        content = response.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            examples = json.loads(content)
            if not isinstance(examples, list):
                raise ValueError("Response is not a list")
            return [str(ex) for ex in examples[:num_samples]]
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: split by newlines
            lines = [line.strip(' "[],-') for line in content.split("\n") if line.strip()]
            return [line for line in lines if line and not line.startswith("//")][:num_samples]


class LLMEvaluator:
    """Evaluate inputs using LLM for semantic rules."""

    def __init__(
        self,
        rule: Rule,
        model: str = DEFAULT_TEST_MODEL,
        cache_mode: CacheMode = CacheMode.PERSISTENT
    ):
        self.rule = rule
        self.model = model
        self.caller = create_caller(
            model=model,
            temperature=0.0,  # Deterministic for evaluation
            cache_mode=cache_mode,
        )

    async def evaluate(self, text: str) -> bool:
        """Evaluate a single input using LLM."""
        prompt = f"""Does the following text match this rule?

Rule: {self.rule.description}

Text: {text}

Answer ONLY with 'true' or 'false' (lowercase, no quotes)."""

        messages = [Message(role="user", content=prompt)]
        response = await self.caller.call(messages)

        # Parse response
        content = response.content.strip().lower()
        matches = re.findall(r"\b(true|false)\b", content)
        unique = set(matches)

        if unique == {"true"}:
            return True
        if unique == {"false"}:
            return False
        if matches:
            first = matches[0]
            if first == "true":
                return True
            if first == "false":
                return False

        raise ValueError(f"Ambiguous LLM evaluation response: {response.content!r}")

    async def evaluate_batch(self, texts: list[str]) -> list[bool]:
        """Evaluate multiple inputs."""
        results = []
        for text in texts:
            result = await self.evaluate(text)
            results.append(result)
        return results


# ============================================================================
# Dataset Generator
# ============================================================================


@dataclass
class GeneratorConfig:
    """Configuration for dataset generation."""
    rules_file: Path
    num_samples: int = 200
    output_dir: Path = Path("experiments/datasets")
    use_llm: bool = False
    models: list[str] = None
    random_seed: int = 42
    balance_ratio: float = 0.5  # Target ratio of positive samples

    def __post_init__(self):
        if self.models is None:
            self.models = [DEFAULT_TEST_MODEL]


class DatasetGenerator:
    """Main dataset generator."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        set_random_seed(config.random_seed)

    def _load_rules(self) -> list[Rule]:
        """Load rules from JSONL file."""
        rules = []
        with self.config.rules_file.open("r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    rules.append(Rule(**data))
        return rules

    async def generate_for_rule(self, rule: Rule) -> tuple[list[DatasetSample], DatasetMetadata]:
        """Generate dataset for a single rule."""
        print(f"\nGenerating dataset for: {rule.name} ({rule.rule_id})")

        # Determine if rule needs LLM evaluation
        needs_llm_eval = rule.name in LLM_EVAL_RULES

        # Determine generation strategy
        use_llm = self.config.use_llm or not rule.programmatic or needs_llm_eval

        if use_llm:
            generator = LLMGenerator(
                rule=rule,
                model=self.config.models[0],
                random_seed=self.config.random_seed
            )
            method = "llm"
            models_used = self.config.models
        else:
            generator = ProgrammaticGenerator(
                rule=rule,
                random_seed=self.config.random_seed
            )
            method = "programmatic"
            models_used = []

        evaluator: Optional[LLMEvaluator] = None
        eval_fn: Optional[Callable[[str], bool]] = None

        if needs_llm_eval:
            print("  Using LLM evaluation for semantic rule...")
            evaluator = LLMEvaluator(rule=rule, model=self.config.models[0])
        else:
            eval_fn = EVAL_FUNCTIONS.get(rule.name)
            if eval_fn is None:
                raise ValueError(f"No evaluation function for rule: {rule.name}")

        async def evaluate_input(text: str) -> bool:
            if evaluator is not None:
                return await evaluator.evaluate(text)
            assert eval_fn is not None
            return eval_fn(text)

        async def collect_valid_samples(
            target_count: int,
            target_label: bool,
        ) -> tuple[list[str], list[dict[str, Any]], int]:
            """Generate samples until we have target_count valid labels."""
            valid: list[str] = []
            errors: list[dict[str, Any]] = []
            attempts = 0
            evaluations = 0
            max_attempts = 10

            while len(valid) < target_count and attempts < max_attempts:
                batch_size = max(target_count - len(valid), 1)
                candidates = await generator.generate(batch_size, target_label)

                for candidate in candidates:
                    try:
                        actual_label = await evaluate_input(candidate)
                        evaluations += 1
                    except ValueError as err:
                        errors.append(
                            {
                                "input": candidate,
                                "expected": target_label,
                                "actual": None,
                                "error": str(err),
                            }
                        )
                        continue

                    if actual_label == target_label:
                        valid.append(candidate)
                    else:
                        errors.append(
                            {
                                "input": candidate,
                                "expected": target_label,
                                "actual": actual_label,
                            }
                        )
                attempts += 1

            if len(valid) < target_count:
                print(
                    f"  WARNING: Only collected {len(valid)}/{target_count} "
                    f"{'positive' if target_label else 'negative'} samples after "
                    f"{attempts} attempts"
                )

            return valid[:target_count], errors, evaluations

        # Calculate target counts
        num_positive = int(self.config.num_samples * self.config.balance_ratio)
        num_negative = self.config.num_samples - num_positive

        print(
            f"  Targeting {num_positive} positive and {num_negative} negative samples..."
        )
        positive_inputs, pos_errors, pos_evals = await collect_valid_samples(
            num_positive, True
        )
        negative_inputs, neg_errors, neg_evals = await collect_valid_samples(
            num_negative, False
        )

        samples = []
        for inp in positive_inputs:
            samples.append(
                DatasetSample(
                    input=inp,
                    label=True,
                    rule_id=rule.rule_id,
                    metadata={"expected_label": True, "correct": True},
                )
            )

        for inp in negative_inputs:
            samples.append(
                DatasetSample(
                    input=inp,
                    label=False,
                    rule_id=rule.rule_id,
                    metadata={"expected_label": False, "correct": True},
                )
            )

        # Shuffle samples
        random.shuffle(samples)

        # Quality checks
        actual_positive = sum(1 for s in samples if s.label)
        actual_negative = len(samples) - actual_positive
        total_evaluated = pos_evals + neg_evals
        total_errors = len(pos_errors) + len(neg_errors)
        accuracy = (
            (total_evaluated - total_errors) / total_evaluated
            if total_evaluated
            else 0.0
        )

        if samples:
            input_lengths = [len(s.input) for s in samples]
            positive_ratio = actual_positive / len(samples)
            input_length_stats = {
                "min": min(input_lengths),
                "max": max(input_lengths),
                "avg": sum(input_lengths) / len(input_lengths),
            }
        else:
            positive_ratio = 0.0
            input_length_stats = {"min": 0, "max": 0, "avg": 0.0}

        quality_checks = {
            "accuracy": accuracy,
            "positive_ratio": positive_ratio,
            "num_errors": total_errors,
            "input_length_stats": input_length_stats,
            "errors": (pos_errors + neg_errors)[:10],  # First 10 errors
        }

        print(f"  Quality: {accuracy*100:.1f}% correct labels")
        print(f"  Actual split: {actual_positive} positive / {actual_negative} negative")
        if total_errors:
            print(f"  WARNING: {total_errors} candidate samples discarded during filtering!")

        # Create metadata
        metadata = DatasetMetadata(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            rule_description=rule.description,
            num_samples=len(samples),
            num_positive=actual_positive,
            num_negative=actual_negative,
            generation_method=method,
            models_used=models_used,
            random_seed=self.config.random_seed,
            timestamp=datetime.now().isoformat(),
            quality_checks=quality_checks,
        )

        return samples, metadata

    async def generate_all(self) -> dict[str, Path]:
        """Generate datasets for all rules."""
        rules = self._load_rules()
        print(f"Loaded {len(rules)} rules from {self.config.rules_file}")

        generated_files = {}
        all_metadata = {}

        for rule in rules:
            samples, metadata = await self.generate_for_rule(rule)

            # Save dataset
            dataset_file = self.config.output_dir / f"{rule.rule_id}.jsonl"
            save_jsonl(
                [s.model_dump() for s in samples],
                dataset_file
            )
            generated_files[rule.rule_id] = dataset_file
            all_metadata[rule.rule_id] = metadata.model_dump()

            print(f"  Saved to: {dataset_file}")

        # Save combined metadata
        metadata_file = self.config.output_dir / "metadata.yaml"
        save_yaml(all_metadata, metadata_file)
        print(f"\nSaved metadata to: {metadata_file}")

        # Print summary
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        for rule_id, meta in all_metadata.items():
            print(f"\n{meta['rule_name']} ({rule_id}):")
            print(f"  Samples: {meta['num_samples']} ({meta['num_positive']}+/{meta['num_negative']}-)")
            print(f"  Method: {meta['generation_method']}")
            print(f"  Accuracy: {meta['quality_checks']['accuracy']*100:.1f}%")
            if meta['quality_checks']['num_errors'] > 0:
                print(f"  Errors: {meta['quality_checks']['num_errors']} ⚠️")

        return generated_files


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> GeneratorConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate fixed datasets for classification rules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--rules-file",
        type=Path,
        required=True,
        help="JSONL file with rule definitions"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of samples per rule"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/datasets"),
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for input generation (even for simple rules)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_TEST_MODEL],
        help="Models to use for LLM generation"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--balance-ratio",
        type=float,
        default=0.5,
        help="Target ratio of positive samples (0.0-1.0)"
    )

    args = parser.parse_args()

    return GeneratorConfig(
        rules_file=args.rules_file,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        use_llm=args.use_llm,
        models=args.models,
        random_seed=args.random_seed,
        balance_ratio=args.balance_ratio,
    )


async def main():
    """Main entry point."""
    config = parse_args()

    generator = DatasetGenerator(config)
    await generator.generate_all()


if __name__ == "__main__":
    asyncio.run(main())
