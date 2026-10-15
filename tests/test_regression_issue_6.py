import json
import importlib
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

emotion_prompt = importlib.import_module("emotion-prompt")


def test_new_topic_is_classified_as_indifference_and_appended_not_refused():
    """
    Regression test for GitHub issue #6:
    Proves that a new topic unknown to get_emotion is classified as 'indifference'
    and then accepted/appended to emotion_map['indifference'] rather than refused.
    """
    new_topic = "cryptography"
    keyword = "ciphers"

    # Verify the topic is not initially known to any emotion
    for emotion, topics in emotion_prompt.emotion_map.items():
        assert new_topic not in topics

    # Step 1: get_emotion classifies new topic as indifference
    emotion = emotion_prompt.get_emotion(new_topic)
    assert emotion == "indifference"

    # Step 2: update_emotion_map appends the new topic to indifference
    emotion_prompt.update_emotion_map(emotion, new_topic, keyword)

    # Verify it is accepted and appended, not refused
    assert new_topic in emotion_prompt.emotion_map["indifference"]

    # Verify get_emotion now returns indifference for the topic from the map
    assert emotion_prompt.get_emotion(new_topic) == "indifference"


def test_prompt_flow_with_new_topic_mocking_network_calls():
    """
    Integration regression test for prompt() flow mocking network/LLM calls:
    Proves that when Ollama returns a new topic and keyword from user prompt,
    the new topic is classified as indifference, accepted, and appended.
    """
    new_topic = "astronomy"
    keyword = "telescope"
    user_input = "I like astronomy and stars"

    # Verify topic is not initially in emotion_map
    for emotion, topics in emotion_prompt.emotion_map.items():
        assert new_topic not in topics

    topic_mock_response = json.dumps({"response": f"{new_topic} {keyword}"})
    emotion_mock_response = json.dumps({"response": "The cosmos is vast."})

    mock_resp_1 = MagicMock()
    mock_resp_1.text = topic_mock_response
    mock_resp_1.raise_for_status = MagicMock()

    mock_resp_2 = MagicMock()
    mock_resp_2.text = emotion_mock_response
    mock_resp_2.raise_for_status = MagicMock()

    with patch("requests.post", side_effect=[mock_resp_1, mock_resp_2]) as mock_post, \
         patch("builtins.input", side_effect=[user_input, "quit"]):
        emotion_prompt.prompt()

    assert mock_post.call_count == 2
    assert new_topic in emotion_prompt.emotion_map["indifference"]
    assert emotion_prompt.get_emotion(new_topic) == "indifference"


def test_update_emotion_map_does_not_duplicate_existing_topic():
    """Verify that updating an already appended topic does not add duplicates."""
    topic = "robotics"
    keyword = "automation"

    emotion_prompt.update_emotion_map("indifference", topic, keyword)
    assert emotion_prompt.emotion_map["indifference"].count(topic) == 1

    emotion_prompt.update_emotion_map("indifference", topic, keyword)
    assert emotion_prompt.emotion_map["indifference"].count(topic) == 1


def test_update_emotion_map_refuses_when_keyword_is_empty_or_none():
    """Verify that update_emotion_map requires a valid keyword."""
    emotion_prompt.update_emotion_map("indifference", "ghost", None)
    assert "ghost" not in emotion_prompt.emotion_map["indifference"]

    emotion_prompt.update_emotion_map("indifference", "ghost", "")
    assert "ghost" not in emotion_prompt.emotion_map["indifference"]


def test_known_topic_flow_mocking_llm_query():
    """Verify that existing known topics remain correctly classified and not duplicated."""
    known_topic = "peace"
    keyword = "harmony"

    assert emotion_prompt.get_emotion(known_topic) == "happiness"
    initial_count = emotion_prompt.emotion_map["happiness"].count(known_topic)

    with patch.object(emotion_prompt, "query_ollama", side_effect=[f"{known_topic} {keyword}", "Peaceful response"]), \
         patch("builtins.input", side_effect=["give me peace", "quit"]):
        emotion_prompt.prompt()

    assert emotion_prompt.emotion_map["happiness"].count(known_topic) == initial_count
