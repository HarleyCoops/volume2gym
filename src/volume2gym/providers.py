"""Provider-neutral structured generation interfaces."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from volume2gym.sources import detect_media_type


class ProviderError(RuntimeError):
    """A durable, user-visible provider failure."""


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    system: str
    prompt: str
    schema_name: str
    images: tuple[Path, ...] = ()
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass(frozen=True, slots=True)
class GenerationResult:
    value: Mapping[str, Any]
    provider: str
    model: str
    raw_text: str


class StructuredGenerator(Protocol):
    provider: str
    model: str

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"provider returned invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ProviderError("provider returned JSON that is not an object")
    return value


class CallableGenerator:
    """Adapter for local functions and deterministic tests."""

    def __init__(
        self,
        function: Callable[[GenerationRequest], str | Mapping[str, Any]],
        *,
        provider: str = "callable",
        model: str = "local",
    ) -> None:
        self.function = function
        self.provider = provider
        self.model = model

    def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            output = self.function(request)
        except Exception as exc:
            raise ProviderError(f"{self.provider}/{self.model} failed: {exc}") from exc
        if isinstance(output, str):
            raw_text = output
            value = parse_json_object(output)
        else:
            value = dict(output)
            raw_text = json.dumps(value, sort_keys=True)
        return GenerationResult(
            value=value,
            provider=self.provider,
            model=self.model,
            raw_text=raw_text,
        )


class AnthropicGenerator:
    """Optional Anthropic implementation; credentials are supplied by the caller."""

    provider = "anthropic"

    def __init__(self, *, api_key: str, model: str, client: Any | None = None) -> None:
        if not api_key and client is None:
            raise ValueError("api_key is required")
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ProviderError(
                    "Anthropic support requires `pip install volume2gym[anthropic]`"
                ) from exc
            client = anthropic.Anthropic(api_key=api_key)
        self.client = client
        self.model = model

    def generate(self, request: GenerationRequest) -> GenerationResult:
        content: list[dict[str, Any]] = []
        for image in request.images:
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": detect_media_type(image),
                        "data": encoded,
                    },
                }
            )
        content.append({"type": "text", "text": request.prompt})
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system,
                messages=[{"role": "user", "content": content}],
            )
            raw_text = message.content[0].text
        except Exception as exc:
            raise ProviderError(f"anthropic/{self.model} failed: {exc}") from exc
        return GenerationResult(
            value=parse_json_object(raw_text),
            provider=self.provider,
            model=self.model,
            raw_text=raw_text,
        )


def encode_images(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    images = tuple(Path(path) for path in paths)
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return images
