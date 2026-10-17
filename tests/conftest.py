import importlib
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_builtin_map(emotion_prompt):
    """Reset in-memory state without reading the on-disk map."""
    emotion_prompt.MODEL = "llama3.2"
    emotion_prompt.OLLAMA_PORT = 11434
    emotion_prompt._emotion_map_source = None
    emotion_prompt._overwrite_blocked.clear()
    fresh = emotion_prompt.builtin_emotion_map()
    emotion_prompt.emotion_map.clear()
    emotion_prompt.emotion_map.update(fresh)


@pytest.fixture(autouse=True)
def reset_emotion_map():
    emotion_prompt = importlib.import_module("emotion-prompt")
    _install_builtin_map(emotion_prompt)
    yield
    _install_builtin_map(emotion_prompt)


@pytest.fixture(autouse=True)
def isolate_emotion_map_path(tmp_path, monkeypatch):
    """Point persistence at a throwaway file for every test.

    The map is loaded lazily, after this redirect, so importing the
    module does not read ~/.emotional-prompt/emotion_map.json.
    """
    emotion_prompt = importlib.import_module("emotion-prompt")
    emotion_prompt.EMOTION_MAP_PATH = tmp_path / "emotion_map.json"
    emotion_prompt._emotion_map_source = None
    monkeypatch.delenv("EMOTION_MAP_PATH", raising=False)
    yield
