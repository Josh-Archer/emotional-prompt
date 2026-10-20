import importlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

emotion_prompt = importlib.import_module("emotion-prompt")


def test_parse_args_defaults():
    """Verify that default model and port are llama3.2 and 11434."""
    args = emotion_prompt.parse_args([])
    assert args.model == "llama3.2"
    assert args.port == 11434


def test_parse_args_long_options():
    """Verify that --model and --port long options are parsed correctly."""
    args = emotion_prompt.parse_args(
        ["--model", "mistral:latest", "--port", "11435"]
    )
    assert args.model == "mistral:latest"
    assert args.port == 11435


def test_parse_args_short_options():
    """Verify that -m and -p short options are parsed correctly."""
    args = emotion_prompt.parse_args(["-m", "gemma:2b", "-p", "8080"])
    assert args.model == "gemma:2b"
    assert args.port == 8080


def test_parse_args_mixed_options():
    """Verify that mixed long and short options work seamlessly."""
    args1 = emotion_prompt.parse_args(["--model", "qwen2.5", "-p", "9000"])
    assert args1.model == "qwen2.5"
    assert args1.port == 9000

    args2 = emotion_prompt.parse_args(["-m", "phi3", "--port", "9100"])
    assert args2.model == "phi3"
    assert args2.port == 9100


def test_parse_args_help(capsys):
    """Verify that --help exits with code 0 and displays options."""
    with pytest.raises(SystemExit) as exc_info:
        emotion_prompt.parse_args(["--help"])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "--model" in captured.out
    assert "-m" in captured.out
    assert "--port" in captured.out
    assert "-p" in captured.out
    assert "llama3.2" in captured.out
    assert "11434" in captured.out


def test_parse_args_invalid_port(capsys):
    """Verify that a non-integer port raises SystemExit with code 2."""
    with pytest.raises(SystemExit) as exc_info:
        emotion_prompt.parse_args(["--port", "invalid_port"])
    assert exc_info.value.code == 2

    captured = capsys.readouterr()
    assert "invalid int value" in captured.err


def test_main_updates_globals_with_cli_args():
    """Verify that main(args) updates MODEL and OLLAMA_PORT globals."""
    with patch("builtins.input", return_value="quit"):
        emotion_prompt.main(["--model", "custom-model", "--port", "12345"])

    assert emotion_prompt.MODEL == "custom-model"
    assert emotion_prompt.OLLAMA_PORT == 12345


def test_main_uses_default_values_when_no_args():
    """Verify that main([]) retains or resets to default values."""
    with patch("builtins.input", return_value="quit"):
        emotion_prompt.main([])

    assert emotion_prompt.MODEL == "llama3.2"
    assert emotion_prompt.OLLAMA_PORT == 11434


def test_query_ollama_uses_custom_model_and_port():
    """Verify query_ollama targets configured port and model in payload."""
    with patch("builtins.input", return_value="quit"):
        emotion_prompt.main(["--model", "custom-llm", "--port", "19999"])

    mock_resp = MagicMock()
    mock_resp.text = '{"response": "test response"}'
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = emotion_prompt.query_ollama("hello world")

    assert res == "test response"
    mock_post.assert_called_once_with(
        "http://127.0.0.1:19999/api/generate",
        json={"model": "custom-llm", "prompt": "hello world", "stream": False},
        timeout=5,
    )


def test_query_ollama_connection_error_prints_custom_model_and_port(capsys):
    """Verify connection error message prints custom port and model."""
    import requests

    with patch("builtins.input", return_value="quit"):
        emotion_prompt.main(["--model", "custom-llm", "--port", "19999"])

    conn_err = requests.exceptions.ConnectionError("Connection refused")
    with patch("requests.post", side_effect=conn_err):
        res = emotion_prompt.query_ollama("test error")

    assert res is None
    captured = capsys.readouterr()
    assert "Could not connect to Ollama on 127.0.0.1:19999" in captured.out
    assert "ollama run custom-llm" in captured.out


def test_cli_subprocess_help():
    """Verify end-to-end CLI execution with --help flag."""
    script_path = REPO_ROOT / "emotion-prompt.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--port" in result.stdout
    assert "llama3.2" in result.stdout
    assert "11434" in result.stdout
