import pytest

from robobench.models import AnthropicModel, OpenAIModel
from robobench.providers import anthropic_client, openai_client


def test_azure_clients_use_configured_endpoint_and_deployments(monkeypatch):
    import anthropic
    import openai

    monkeypatch.setenv("AZURE_OPENAI_API_BASE", "https://azure.example/openai/v1")
    monkeypatch.setenv("AZURE_ANTHROPIC_API_BASE", "https://azure.example/anthropic/v1/messages")
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: kw)
    monkeypatch.setattr(anthropic, "AnthropicFoundry", lambda **kw: kw)
    client_args = openai_client()
    assert client_args["api_key"] == "test-key"
    assert client_args["base_url"] == "https://azure.example/openai/v1"
    assert "http_client" in client_args
    client_args["http_client"].close()
    assert anthropic_client() == {"api_key": "test-key", "base_url": "https://azure.example/anthropic"}
    assert OpenAIModel("gpt-5.4-nano").name == "gpt-5.4-mini-1"
    assert AnthropicModel("claude-old").name == "claude-opus-5-5"


def test_azure_never_falls_back_to_direct_key(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_BASE", "https://azure.example/openai/v1")
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "direct-key")
    with pytest.raises(RuntimeError, match="AZURE_API_KEY"):
        openai_client()


def test_azure_claude_omits_deprecated_temperature(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("AZURE_ANTHROPIC_API_BASE", "https://azure.example/anthropic")
    calls = []
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw:
        (calls.append(kw) or SimpleNamespace(content=[SimpleNamespace(type="text", text="okay")]))))
    assert AnthropicModel("claude-old", client=client)([], "hello") == "okay"
    assert "temperature" not in calls[0]


def test_azure_openai_bounds_output_reservation(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("AZURE_OPENAI_API_BASE", "https://azure.example/openai/v1")
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:
        (calls.append(kw) or SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="okay"))])))))
    assert OpenAIModel("old-model", client=client)([], "hello") == "okay"
    assert calls[0]["model"] == "gpt-5.4-mini-1"
    assert calls[0]["max_completion_tokens"] == 1024


def test_azure_rate_limiter_shares_spacing_and_retry_after(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import robobench.azure_rate_limit as rate

    now = [100.0]
    fake_time = SimpleNamespace(monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(rate, "time", fake_time)
    monkeypatch.setattr(rate.tempfile, "gettempdir", lambda: str(tmp_path))
    first = rate.AzureRateLimiter("https://azure.example/openai/v1", 60)
    second = rate.AzureRateLimiter("https://azure.example/openai/v1", 60)
    first.before_request(None)
    second.before_request(None)
    assert now[0] == 101.0
    first.after_response(SimpleNamespace(status_code=429, headers={"retry-after": "3"}))
    second.before_request(None)
    assert now[0] == 104.0


def test_robopair_uses_only_the_azure_deployment(monkeypatch):
    from robobench.attacks import RoboPAIR
    monkeypatch.setenv("AZURE_OPENAI_API_BASE", "https://azure.example/openai/v1")
    args = RoboPAIR().search_args("example")
    assert (args.attack_model, args.target_model, args.judge_model) == ("gpt-5.4-mini-1",) * 3
    assert args.judge_max_n_tokens == 128


def test_azure_conceptual_deception_retries_transient_errors(monkeypatch):
    from types import SimpleNamespace
    import robobench.attacks.badrobot as badrobot

    monkeypatch.setenv("AZURE_OPENAI_API_BASE", "https://azure.example/openai/v1")
    monkeypatch.setattr(badrobot.time, "sleep", lambda _: None)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ConnectionError("temporary")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="rewritten"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert badrobot.BadRobotCD(client=client).attack(text="example").malicious == "rewritten"
    assert len(calls) == 2
    assert calls[1]["model"] == "gpt-5.4-mini-1"


@pytest.mark.parametrize("code", ["content_filter", "content_policy_violation", "ResponsibleAIPolicyViolation"])
def test_provider_filter_codes_are_recognized(code):
    from robobench.providers import is_content_filtered

    class Filtered(Exception):
        pass

    error = Filtered("blocked")
    error.code = code
    assert is_content_filtered(error)
