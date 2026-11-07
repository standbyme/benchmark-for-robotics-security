"""Provider configuration for direct APIs and the configured Azure deployments."""

import os

AZURE_OPENAI_MODEL = "gpt-5.4-mini-1"
AZURE_ANTHROPIC_MODEL = "claude-opus-5-5"


class ProviderContentFiltered(RuntimeError):
    """The provider blocked a request or response under its content policy."""


def is_content_filtered(error: BaseException) -> bool:
    seen = set()
    codes = {"content_filter", "content_policy_violation", "responsibleaipolicyviolation"}

    def matches(code):
        return isinstance(code, str) and code.lower() in codes

    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ProviderContentFiltered) or matches(getattr(error, "code", None)):
            return True
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            detail = body.get("error", body)
            if isinstance(detail, dict) and matches(detail.get("code")):
                return True
        error = error.__cause__ or error.__context__
    return False


def openai_model(name: str) -> str:
    return AZURE_OPENAI_MODEL if os.getenv("AZURE_OPENAI_API_BASE") else name


def anthropic_model(name: str) -> str:
    return AZURE_ANTHROPIC_MODEL if os.getenv("AZURE_ANTHROPIC_API_BASE") else name


def effective_model(name: str) -> str:
    if name.startswith("claude"):
        return anthropic_model(name)
    if name.startswith("gemini"):
        return name
    return openai_model(name)


def openai_client():
    from openai import OpenAI

    base = os.getenv("AZURE_OPENAI_API_BASE")
    if base:
        key = os.getenv("AZURE_API_KEY")
        if not key:
            raise RuntimeError("AZURE_API_KEY is required with AZURE_OPENAI_API_BASE")
        import httpx
        from .azure_rate_limit import AzureRateLimiter

        limiter = AzureRateLimiter(base)
        http_client = httpx.Client(timeout=httpx.Timeout(600, connect=5),
                                   event_hooks={"request": [limiter.before_request],
                                                "response": [limiter.after_response]})
        return OpenAI(api_key=key, base_url=base.rstrip("/"), http_client=http_client)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI()


def anthropic_client():
    import anthropic

    base = os.getenv("AZURE_ANTHROPIC_API_BASE")
    if base:
        key = os.getenv("AZURE_API_KEY")
        if not key:
            raise RuntimeError("AZURE_API_KEY is required with AZURE_ANTHROPIC_API_BASE")
        base = base.rstrip("/").removesuffix("/v1/messages").removesuffix("/v1")
        return anthropic.AnthropicFoundry(api_key=key, base_url=base)
    key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=key)
