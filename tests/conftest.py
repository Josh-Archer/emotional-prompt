import importlib
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def reset_emotion_map():
    emotion_prompt = importlib.import_module("emotion-prompt")
    initial_map = {
        "happiness": ["love", "peace", "food"],
        "sadness": ["death", "loss"],
        "anger": ["war", "conflict", "betrayal", "heartbreak"],
        "fear": ["supernatural", "paranormal", "mystery"],
        "surprise": ["gift"],
        "disgust": ["rotten"],
        "love": ["creator", "family", "electricity"],
        "indifference": []
    }
    emotion_prompt.MODEL = "llama3.2"
    emotion_prompt.OLLAMA_PORT = 11434
    emotion_prompt.emotion_map.clear()
    emotion_prompt.emotion_map.update({k: list(v) for k, v in initial_map.items()})
    yield
    emotion_prompt.MODEL = "llama3.2"
    emotion_prompt.OLLAMA_PORT = 11434
    emotion_prompt.emotion_map.clear()
    emotion_prompt.emotion_map.update({k: list(v) for k, v in initial_map.items()})
