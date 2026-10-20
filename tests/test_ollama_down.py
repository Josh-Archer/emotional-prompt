import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "emotion-prompt.py"


def load_module():
    """Load emotion-prompt.py using importlib."""
    spec = importlib.util.spec_from_file_location(
        "emotion_prompt", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


emotion_prompt = load_module()


def test_topic_keyword_query_ollama_returns_none_when_query_ollama_is_none():
    """Return None without raising when query_ollama returns None."""
    with patch.object(
        emotion_prompt, "query_ollama", return_value=None
    ) as mock_query:
        result = emotion_prompt.topic_keyword_query_ollama("test prompt")
        assert result is None
        mock_query.assert_called_once()


def test_topic_keyword_query_ollama_does_not_raise_attribute_error():
    """Do not raise AttributeError when query_ollama returns None."""
    with patch.object(emotion_prompt, "query_ollama", return_value=None):
        try:
            res = emotion_prompt.topic_keyword_query_ollama("tell me a story")
            assert res is None
        except AttributeError:
            pytest.fail(
                "AttributeError was raised when query_ollama returned None"
            )


def test_prompt_shows_connection_message_when_ollama_down(capsys):
    """Show the connection message when Ollama is down."""
    with (
        patch.object(emotion_prompt, "query_ollama", return_value=None),
        patch("builtins.input", side_effect=["hello", "quit"]),
    ):
        emotion_prompt.prompt()

    captured = capsys.readouterr()
    assert "Could not determine the topic. Is Ollama running?" in captured.out


def test_topic_keyword_query_ollama_succeeds_when_query_ollama_returns_topic():
    """Keep working normally when Ollama responds."""
    with patch.object(
        emotion_prompt, "query_ollama", return_value="happiness joy"
    ):
        result = emotion_prompt.topic_keyword_query_ollama("test prompt")
        assert result == {"topic": "happiness", "keyword": "joy"}


def test_topic_keyword_query_ollama_with_monkeypatch(monkeypatch):
    """Test with pytest's monkeypatch fixture reloading via importlib."""
    mod = load_module()
    monkeypatch.setattr(mod, "query_ollama", lambda prompt: None)
    result = mod.topic_keyword_query_ollama("sample prompt")
    assert result is None


def test_topic_keyword_query_ollama_when_requests_connection_error():
    """Test integration when requests raises ConnectionError."""
    import requests

    with patch(
        "requests.post",
        side_effect=requests.exceptions.ConnectionError("Connection refused"),
    ):
        result = emotion_prompt.topic_keyword_query_ollama("hello world")
        assert result is None
