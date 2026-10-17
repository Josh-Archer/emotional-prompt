import importlib
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

emotion_prompt = importlib.import_module("emotion-prompt")


def test_retry_succeeds_on_first_attempt():
    """Verify that valid 2-token reply succeeds without retrying."""
    with patch.object(
        emotion_prompt, "query_ollama", return_value="happiness joy"
    ) as mock_query:
        result = emotion_prompt.topic_keyword_query_ollama("happy day")

    assert result == {"topic": "happiness", "keyword": "joy"}
    assert mock_query.call_count == 1


def test_retry_succeeds_after_initial_failures():
    """Verify that function retries when initial responses lack 2 tokens."""
    responses = [
        "happiness",  # 1 token (fails)
        "happiness joy peace",  # 3 tokens (fails)
        "anger war",  # 2 tokens (succeeds)
    ]
    with patch.object(
        emotion_prompt, "query_ollama", side_effect=responses
    ) as mock_query:
        result = emotion_prompt.topic_keyword_query_ollama("battlefield")

    assert result == {"topic": "anger", "keyword": "war"}
    assert mock_query.call_count == 3


def test_retry_fails_after_max_retries_and_raises_clear_error():
    """Verify bounded retry and clear ValueError when all attempts fail."""
    with patch.object(
        emotion_prompt, "query_ollama", return_value="singleword"
    ) as mock_query:
        with pytest.raises(ValueError) as exc_info:
            emotion_prompt.topic_keyword_query_ollama("some prompt")

    assert mock_query.call_count == 3
    error_msg = str(exc_info.value)
    assert "two tokens" in error_msg
    assert "after 3 attempts" in error_msg
    assert "'singleword'" in error_msg


def test_retry_fails_with_too_many_tokens_message():
    """Verify error message when model returns too many words."""
    with patch.object(
        emotion_prompt, "query_ollama", return_value="one two three four"
    ) as mock_query:
        with pytest.raises(ValueError) as exc_info:
            emotion_prompt.topic_keyword_query_ollama("too long")

    assert mock_query.call_count == 3
    error_msg = str(exc_info.value)
    assert "two tokens" in error_msg
    assert "after 3 attempts" in error_msg
    assert "'one two three four'" in error_msg


def test_clear_error_printed_to_console(capsys):
    """Verify that an Error is printed to console on retry exhaustion."""
    with patch.object(
        emotion_prompt, "query_ollama", return_value="onlyone"
    ):
        with pytest.raises(ValueError):
            emotion_prompt.topic_keyword_query_ollama("test")

    captured = capsys.readouterr()
    expected = "Error: Model did not return two tokens after 3 attempts"
    assert expected in captured.out


def test_custom_max_retries_parameter():
    """Verify custom max_retries parameter bounds attempts accordingly."""
    with patch.object(
        emotion_prompt, "query_ollama", return_value="invalid"
    ) as mock_query:
        with pytest.raises(ValueError) as exc_info:
            emotion_prompt.topic_keyword_query_ollama(
                "custom retries", max_retries=5
            )

    assert mock_query.call_count == 5
    assert "after 5 attempts" in str(exc_info.value)


def test_custom_max_retries_singular_attempt():
    """Verify singular 'attempt' phrasing when max_retries is 1."""
    with patch.object(
        emotion_prompt, "query_ollama", return_value="invalid"
    ) as mock_query:
        with pytest.raises(ValueError) as exc_info:
            emotion_prompt.topic_keyword_query_ollama(
                "single retry", max_retries=1
            )

    assert mock_query.call_count == 1
    assert "after 1 attempt " in str(exc_info.value)


def test_retries_alias_parameter():
    """Verify retries keyword argument works as alias for max_retries."""
    with patch.object(
        emotion_prompt, "query_ollama", return_value="bad"
    ) as mock_query:
        with pytest.raises(ValueError) as exc_info:
            emotion_prompt.topic_keyword_query_ollama(
                "alias test", retries=2
            )

    assert mock_query.call_count == 2
    assert "after 2 attempts" in str(exc_info.value)


def test_invalid_max_retries_raises_value_error():
    """Verify non-positive max_retries raises ValueError."""
    with patch.object(emotion_prompt, "query_ollama") as mock_query:
        with pytest.raises(ValueError, match="must be greater than 0"):
            emotion_prompt.topic_keyword_query_ollama("invalid", max_retries=0)
        with pytest.raises(ValueError, match="must be greater than 0"):
            emotion_prompt.topic_keyword_query_ollama(
                "invalid", max_retries=-2
            )
        with pytest.raises(ValueError, match="must be greater than 0"):
            emotion_prompt.topic_keyword_query_ollama(
                "invalid", max_retries=None
            )

    assert mock_query.call_count == 0


def test_query_ollama_none_does_not_retry():
    """Verify query_ollama returning None aborts without retrying."""
    with patch.object(
        emotion_prompt, "query_ollama", return_value=None
    ) as mock_query:
        result = emotion_prompt.topic_keyword_query_ollama("test")

    assert result is None
    assert mock_query.call_count == 1


def test_query_ollama_none_during_retry_aborts():
    """Verify query_ollama returning None mid-retry returns None."""
    responses = ["invalid_word", None]
    with patch.object(
        emotion_prompt, "query_ollama", side_effect=responses
    ) as mock_query:
        result = emotion_prompt.topic_keyword_query_ollama("test")

    assert result is None
    assert mock_query.call_count == 2


def test_prompt_recovers_after_failed_attempt(capsys):
    """Verify prompt() handles recovery when model succeeds on retry."""
    with patch.object(
        emotion_prompt,
        "query_ollama",
        side_effect=["one_token", "peace joy", "I am feeling peaceful!"],
    ), patch("builtins.input", side_effect=["hello", "quit"]):
        emotion_prompt.prompt()

    captured = capsys.readouterr()
    assert "Topic: peace" in captured.out
    assert "Emotion: happiness" in captured.out
    assert "Response: I am feeling peaceful!" in captured.out
    assert "Goodbye!" in captured.out


def test_prompt_handles_exhausted_retries_gracefully(capsys):
    """Verify prompt() prints error and does not crash on exhaustion."""
    with patch.object(
        emotion_prompt, "query_ollama", return_value="invalid_token"
    ), patch("builtins.input", side_effect=["failed prompt", "quit"]):
        emotion_prompt.prompt()

    captured = capsys.readouterr()
    expected = "Error: Model did not return two tokens after 3 attempts"
    assert expected in captured.out
    assert "Goodbye!" in captured.out
    # Ollama is running, so connection error message should not be shown
    not_expected = "Could not determine the topic. Is Ollama running?"
    assert not_expected not in captured.out
