"""
Async API caller with intelligent caching for LLM experiments.

Supports:
- OpenAI (GPT-4o-mini)
- Anthropic (Claude 3.5 Haiku)
- Intelligent caching (15-minute or persistent)
- Rate limiting and retry logic
- Proper error handling without broad try/except
"""

import asyncio
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import anthropic
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm_asyncio

# Load environment variables
load_dotenv()

# Import model registry functions
from src.model_registry import requires_new_api

# Global defaults
DEFAULT_TEMPERATURE: float = 1.0


class CacheMode(str, Enum):
    """Cache duration modes."""
    NONE = "none"
    SHORT = "short"  # 15 minutes
    PERSISTENT = "persistent"  # Forever


class Message(BaseModel):
    """Structured message format."""
    role: Literal["system", "user", "assistant"]
    content: str

    def openai_format(self) -> dict[str, str]:
        """Convert to OpenAI message format."""
        return {"role": self.role, "content": self.content}

    def anthropic_format(self) -> dict[str, str]:
        """Convert to Anthropic message format."""
        # Anthropic uses same format
        return {"role": self.role, "content": self.content}


class APIResponse(BaseModel):
    """Standardized API response."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False
    cache_timestamp: Optional[float] = None


@dataclass
class APIConfig:
    """Configuration for API calls."""
    model: str
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = 4096
    cache_mode: CacheMode = CacheMode.SHORT
    cache_dir: Path = Path(".cache")
    max_concurrent: int = 10  # Rate limiting


class HashableBaseModel(BaseModel):
    """Base model with deterministic hashing for caching."""

    def deterministic_hash(self) -> str:
        """Generate deterministic hash from model fields."""
        # Pydantic v2 doesn't support sort_keys in model_dump_json
        # So we dump to dict, then manually serialize with sorted keys
        data = self.model_dump(exclude_none=True)
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


class CacheKey(HashableBaseModel):
    """Cache key for API requests."""
    model: str
    messages: list[Message]
    temperature: float
    max_tokens: int

    def to_hash(self) -> str:
        """Generate cache key hash."""
        return self.deterministic_hash()


class CacheEntry(BaseModel):
    """Cache entry with metadata."""
    response: APIResponse
    timestamp: float
    cache_mode: CacheMode


class APICallerBase(ABC):
    """Abstract base class for API callers."""

    def __init__(self, config: APIConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self._setup_client()

    @abstractmethod
    def _setup_client(self) -> None:
        """Initialize the API client."""
        pass

    @abstractmethod
    async def _call_api(
        self, messages: list[Message], temperature: float, max_tokens: int
    ) -> APIResponse:
        """Make the actual API call."""
        pass

    def _get_cache_path(self, cache_key: CacheKey) -> Path:
        """Get cache file path for a request."""
        key_hash = cache_key.to_hash()
        return self.config.cache_dir / f"{key_hash}.json"

    def _is_cache_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        if entry.cache_mode == CacheMode.PERSISTENT:
            return True
        if entry.cache_mode == CacheMode.SHORT:
            age = time.time() - entry.timestamp
            return age < 15 * 60  # 15 minutes
        return False

    def _load_from_cache(self, cache_key: CacheKey) -> Optional[APIResponse]:
        """Load response from cache if valid."""
        if self.config.cache_mode == CacheMode.NONE:
            return None

        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            with cache_path.open("r") as f:
                entry = CacheEntry.model_validate_json(f.read())

            if self._is_cache_valid(entry):
                response = entry.response
                response.cached = True
                response.cache_timestamp = entry.timestamp
                return response
        except (json.JSONDecodeError, ValueError):
            # Invalid cache file, ignore
            pass

        return None

    def _save_to_cache(self, cache_key: CacheKey, response: APIResponse) -> None:
        """Save response to cache."""
        if self.config.cache_mode == CacheMode.NONE:
            return

        cache_path = self._get_cache_path(cache_key)
        entry = CacheEntry(
            response=response,
            timestamp=time.time(),
            cache_mode=self.config.cache_mode,
        )

        with cache_path.open("w") as f:
            f.write(entry.model_dump_json(indent=2))

    async def call(
        self,
        messages: list[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> APIResponse:
        """
        Call API with caching and rate limiting.

        Args:
            messages: List of messages to send
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            APIResponse with content and metadata
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        cache_key = CacheKey(
            model=self.config.model,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
        )

        # Check cache
        cached_response = self._load_from_cache(cache_key)
        if cached_response is not None:
            return cached_response

        # Rate limiting
        async with self.semaphore:
            response = await self._call_api(messages, temp, max_tok)

        # Save to cache
        self._save_to_cache(cache_key, response)

        return response

    async def call_batch(
        self,
        batch_messages: list[list[Message]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        show_progress: bool = True,
    ) -> list[APIResponse]:
        """
        Call API for batch of message lists with progress tracking.

        Args:
            batch_messages: List of message lists
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            show_progress: Show progress bar

        Returns:
            List of APIResponse objects
        """
        tasks = [
            self.call(messages, temperature, max_tokens)
            for messages in batch_messages
        ]

        if show_progress:
            responses = []
            for coro in tqdm_asyncio.as_completed(
                tasks, total=len(tasks), desc=f"Calling {self.config.model}"
            ):
                responses.append(await coro)
            return responses
        else:
            return await asyncio.gather(*tasks)


class OpenAICaller(APICallerBase):
    """OpenAI API caller (GPT-4o-mini)."""

    def _setup_client(self) -> None:
        """Initialize OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = openai.AsyncOpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(openai.RateLimitError),
    )
    async def _call_api(
        self, messages: list[Message], temperature: float, max_tokens: int
    ) -> APIResponse:
        """Make OpenAI API call with retry logic."""
        # Newer models (gpt-4.1+, gpt-5+) have different API requirements
        is_new_model = requires_new_api(self.config.model)

        if is_new_model:
            # Newer responses API models: use max_completion_tokens
            kwargs = {
                "model": self.config.model,
                "messages": [msg.openai_format() for msg in messages],
                "max_completion_tokens": max_tokens,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
        else:
            # Legacy chat/completions API models
            kwargs = {
                "model": self.config.model,
                "messages": [msg.openai_format() for msg in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        response = await self.client.chat.completions.create(**kwargs)

        return APIResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )


class AnthropicCaller(APICallerBase):
    """Anthropic API caller (Claude 3.5 Haiku)."""

    def _setup_client(self) -> None:
        """Initialize Anthropic client."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(anthropic.RateLimitError),
    )
    async def _call_api(
        self, messages: list[Message], temperature: float, max_tokens: int
    ) -> APIResponse:
        """Make Anthropic API call with retry logic."""
        # Separate system message from other messages
        system_msg = None
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                chat_messages.append(msg.anthropic_format())

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_msg:
            kwargs["system"] = system_msg

        response = await self.client.messages.create(**kwargs)

        return APIResponse(
            content=response.content[0].text if response.content else "",
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )


def create_caller(
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 4096,
    cache_mode: CacheMode = CacheMode.SHORT,
    cache_dir: Path = Path(".cache"),
    max_concurrent: int = 10,
) -> APICallerBase:
    """
    Factory function to create appropriate API caller.

    Args:
        model: Model name (e.g., "gpt-4o-mini", "claude-3-5-haiku-20241022")
        temperature: Sampling temperature
        max_tokens: Maximum completion tokens
        cache_mode: Cache duration mode
        cache_dir: Directory for cache files
        max_concurrent: Maximum concurrent requests

    Returns:
        Appropriate APICallerBase subclass instance
    """
    config = APIConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        cache_mode=cache_mode,
        cache_dir=cache_dir,
        max_concurrent=max_concurrent,
    )

    if "gpt" in model.lower():
        return OpenAICaller(config)
    elif "claude" in model.lower():
        return AnthropicCaller(config)
    else:
        raise ValueError(f"Unsupported model: {model}")


# Example usage
async def example_usage():
    """Example of using the API caller."""
    # Create caller for GPT-4o-mini with 15-minute cache
    caller = create_caller(
        model="gpt-4o-mini",
        temperature=0.0,
        cache_mode=CacheMode.SHORT,
    )

    # Single call
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is 2+2?"),
    ]
    response = await caller.call(messages)
    print(f"Response: {response.content}")
    print(f"Cached: {response.cached}")

    # Batch call
    batch = [
        [Message(role="user", content=f"What is {i}+{i}?")]
        for i in range(5)
    ]
    responses = await caller.call_batch(batch)
    for i, resp in enumerate(responses):
        print(f"Response {i}: {resp.content}, cached={resp.cached}")


if __name__ == "__main__":
    asyncio.run(example_usage())
