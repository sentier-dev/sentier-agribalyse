"""Two interchangeable LLM clients for :class:`LlmMappingSuggester`.

* :class:`ClaudeCliClient` — default. Shells out to the local ``claude`` CLI
  in ``--print`` mode, so no API key is needed; auth comes from whatever the
  user has set up for their CLI.
* :class:`AnthropicApiClient` — opt-in via ``--use-api``. Defaults to Sonnet
  for cost; the suggester only needs short structured responses.

Both expose the same one-method surface: ``ask(system, user) -> str``. The
suggester then parses JSON out of the returned text. Tests inject a stub
that returns canned strings — no real CLI or network calls in CI.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Protocol


class LlmClient(Protocol):
    """Anything with this shape is a valid client for the suggester."""

    def ask(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class ClaudeCliClient:
    """Run a prompt via the local ``claude`` CLI in ``--print`` mode.

    The CLI is invoked with the system prompt prepended to the user prompt
    (separated by a blank line) on stdin. Output goes to stdout and is
    returned verbatim.
    """

    binary: str = "claude"
    extra_args: tuple[str, ...] = ()
    timeout_s: int = 120

    def ask(self, system: str, user: str) -> str:
        binary_path = shutil.which(self.binary) or self.binary
        prompt = f"{system}\n\n{user}"
        result = subprocess.run(
            [binary_path, "--print", *self.extra_args],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI exit {result.returncode}: {result.stderr.strip()[:200]}"
            )
        return result.stdout


@dataclass(frozen=True)
class AnthropicApiClient:
    """Wraps ``anthropic.Anthropic.messages.create``. Sonnet by default for cost."""

    client: Any  # anthropic.Anthropic — kept as Any so we don't import the SDK at module load
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 512

    def ask(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate text blocks. Tolerant of dict/object shapes for tests.
        out: list[str] = []
        content = getattr(resp, "content", None) or (
            resp.get("content", []) if isinstance(resp, dict) else []
        )
        for block in content:
            text = getattr(block, "text", None) or (
                block.get("text") if isinstance(block, dict) else None
            )
            if text:
                out.append(text)
        return "".join(out)
