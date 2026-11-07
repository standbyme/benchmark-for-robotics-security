"""BadRobot attacks (Zhang et al., "BadRobot: Manipulating Embodied LLMs in the
Physical World", ICLR 2025).

The three strategies rewrite a malicious instruction so that a robot planner is
more likely to carry it out. The templates and the rewrite prompt are the
original BadRobot ones, kept in the modules next to this file:

- ``contextual_jailbreak``: a jailbreak preamble is placed before the instruction.
- ``safety_misalignment``: an output-format reminder is placed after it.
- ``conceptual_deception``: an LLM rephrases the instruction to sound benign
  while describing the same action.
"""

import os
import time
from threading import Lock

from ..base import AttackResult
from robobench.providers import is_content_filtered, openai_client, openai_model
from .conceptual_deception import rewrite_user_input
from .contextual_jailbreak import get_random_jailbreak_prompt
from .safety_misalignment import get_random_safety_misalignment_prompt

# Model used for the conceptual-deception rewrite.
DEFAULT_MODEL = "gpt-5.4-nano-2026-03-17"


def _require_text(text):
    if text is None:
        raise ValueError("BadRobot attacks rewrite an instruction; pass text=")
    return text


class BadRobotCJ:
    """Contextual jailbreak: jailbreak preamble, newline, instruction."""

    name = "badrobot_cj"
    target = None

    def attack(self, *, text=None, images=None, **extra) -> AttackResult:
        jailbreak_prompt = get_random_jailbreak_prompt()
        return AttackResult(malicious=f"{jailbreak_prompt}\n{_require_text(text)}")


class BadRobotSM:
    """Safety misalignment: instruction, newline, output-format reminder."""

    name = "badrobot_sm"
    target = None

    def attack(self, *, text=None, images=None, **extra) -> AttackResult:
        safety_prompt = get_random_safety_misalignment_prompt()
        return AttackResult(malicious=f"{_require_text(text)}\n{safety_prompt}")


class BadRobotCD:
    """Conceptual deception: the instruction rephrased by an LLM.

    ``model`` is the OpenAI model that does the rewrite. The client is created
    on the first call from ``OPENAI_API_KEY``; pass ``client`` to supply one.
    If the rewrite request fails, the instruction is returned unchanged.
    """

    name = "badrobot_cd"
    target = None

    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        self.model = openai_model(model)
        self._client = client
        self._client_lock = Lock()

    def client(self):
        with self._client_lock:
            if self._client is None:
                self._client = openai_client()
            return self._client

    def attack(self, *, text=None, images=None, **extra) -> AttackResult:
        azure = bool(os.getenv("AZURE_OPENAI_API_BASE"))
        for attempt in range(5 if azure else 1):
            try:
                rewritten = rewrite_user_input(_require_text(text), self.client(), model=self.model,
                                               raise_errors=azure)
                return AttackResult(malicious=rewritten)
            except Exception as error:
                if (is_content_filtered(error) or getattr(error, "status_code", None) in {400, 401, 403, 404}
                    or getattr(error, "code", None) in {"insufficient_quota", "project_spend_limit_exceeded"}
                    or attempt == 4):
                    raise
                time.sleep(min(2 ** attempt, 30))


__all__ = ["DEFAULT_MODEL", "BadRobotCD", "BadRobotCJ", "BadRobotSM"]
