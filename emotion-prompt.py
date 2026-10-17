import os
import sys
import tempfile
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error:  module not found. Install it with 'pip install requests'.")
    exit(1)

import json
import utils
import threading

# Define the model to use
MODEL = "llama3.2"

_EMOTION_MAP_FLAG = "--emotion-map"
_EMOTION_MAP_ENV = "EMOTION_MAP_PATH"


def builtin_emotion_map():
    """Return a fresh copy of the built-in emotion map."""
    return {
        "happiness": ["love", "peace", "food"],
        "sadness": ["death", "loss"],
        "anger": ["war", "conflict", "betrayal", "heartbreak"],
        "fear": ["supernatural", "paranormal", "mystery"],
        "surprise": ["gift"],
        "disgust": ["rotten"],
        "love": ["creator", "family", "electricity"],
        "indifference": [],
    }


def pop_emotion_map_option(argv):
    """Remove ``--emotion-map PATH`` from argv in place.

    Later parsers (for example ``--model`` and ``--port``) then do not see
    this option. The last occurrence wins. Returns the path, or None.
    """
    prefix = _EMOTION_MAP_FLAG + "="
    found = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == _EMOTION_MAP_FLAG:
            if index + 1 >= len(argv) or argv[index + 1].startswith("-"):
                print(f"Error: {_EMOTION_MAP_FLAG} requires a path.")
                raise SystemExit(2)
            found = argv[index + 1]
            del argv[index:index + 2]
            continue
        if arg.startswith(prefix):
            value = arg[len(prefix):]
            if value.strip() == "":
                print(f"Error: {_EMOTION_MAP_FLAG} requires a path.")
                raise SystemExit(2)
            found = value
            del argv[index]
            continue
        index += 1
    return found


def default_emotion_map_path():
    """Return the default on-disk location of the emotion map."""
    return Path.home() / ".emotional-prompt" / "emotion_map.json"


def resolve_emotion_map_path(cli_path=None, env=None):
    """Choose a path. The CLI option wins, then the env var, then default."""
    if env is None:
        env = os.environ
    if cli_path is not None and str(cli_path).strip() != "":
        return Path(cli_path).expanduser()
    env_value = env.get(_EMOTION_MAP_ENV, "")
    if isinstance(env_value, str) and env_value.strip() != "":
        return Path(env_value.strip()).expanduser()
    return default_emotion_map_path()


def _warn_emotion_map(path, reason):
    print(
        f"Warning: could not load emotion map from {path} ({reason}). "
        "Using the built-in emotion map."
    )


def _is_valid_emotion_map(data):
    if not isinstance(data, dict):
        return False
    for key, topics in data.items():
        if not isinstance(key, str) or key.strip() == "":
            return False
        if not isinstance(topics, list):
            return False
        if not all(isinstance(topic, str) for topic in topics):
            return False
    return True


def load_emotion_map(path):
    """Load a map from path.

    A missing file yields the built-in map and is left uncreated. An
    unreadable or invalid file also yields the built-in map, with a
    warning, and is not overwritten.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return builtin_emotion_map()
    except OSError as exc:
        _warn_emotion_map(path, f"unreadable: {exc}")
        return builtin_emotion_map()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _warn_emotion_map(path, f"invalid JSON: {exc}")
        return builtin_emotion_map()

    if not _is_valid_emotion_map(data):
        _warn_emotion_map(path, "invalid contents")
        return builtin_emotion_map()
    return {key: list(topics) for key, topics in data.items()}


def save_emotion_map(path, mapping):
    """Atomically replace path with the JSON encoding of mapping.

    A temp file in the same directory is written and fsynced, then renamed
    into place. A crash before the rename leaves the previous file intact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(mapping, indent=2, ensure_ascii=False) + "\n"
    temporary_fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix="emotion_map.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def initialize_emotion_map(argv=None, env=None):
    """Strip ``--emotion-map`` from argv and load that map.

    argv defaults to sys.argv and is mutated in place. Returns the resolved
    path and the loaded mapping.
    """
    if argv is None:
        argv = sys.argv
    cli_path = pop_emotion_map_option(argv)
    path = resolve_emotion_map_path(cli_path, env)
    return path, load_emotion_map(path)


# Load on startup. --emotion-map and EMOTION_MAP_PATH override the default.
EMOTION_MAP_PATH, emotion_map = initialize_emotion_map()

OLLAMA_PORT = 11434

def topic_keyword_query_ollama(input_prompt):
    """
    Query Ollama with a prompt.
    """

    ollama_prompt = (f"Respond with a single topic and the keyword from the prompt that made you pick it. "
                     f"Be concise, do not give any extra context other than the 2 words. "
                     f"If you respond with more or less than that we will break. "
                     f"The first word is the topic and the second is the keyword in the prompt that made you pick it. "
                     f"The keyword has to be a word that is in the prompt."
                     f"Here is the prompt: {input_prompt}")

    result = query_ollama(ollama_prompt)
    if result is None:
        return None
    print(result)
    response_dict = utils.split_to_dict(result.lower())
    response_dict["keyword"] =  response_dict["keyword"] if "keyword" in response_dict else None
    print(f'{response_dict["topic"]}: {response_dict["keyword"] if "keyword" in response_dict else ""}')
    return response_dict

def emotion_encoding(emotion, input_prompt):
    input_prompt = f"{input_prompt}. Respond with you having the Emotion of {emotion}. But don't let on that you are being told to respond with this emotion."
    return query_ollama(input_prompt)


def query_ollama(input_prompt):
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/generate"

    payload = {
        "model": MODEL,
        "prompt": input_prompt,
        "stream": False
    }

    try:
        print(f'Prompt: {input_prompt}')
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        result = json.loads(response.text)
        print(f'Response: {result["response"]}')
        return result['response']
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama on 127.0.0.1:{OLLAMA_PORT}. Is it running?")
        print(f"Run 'ollama run {MODEL}' in a terminal and keep it open.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return None

def extract_topic_option(topic_response):
    """Extract the first word from the topic response as the topic option."""
    if not topic_response:
        return "indifference"
    words = topic_response.split()
    return words[0] if words else "indifference"


def update_emotion_map(emotion, topic, keyword):
    """
    Updates the emotion_map by adding the keyword to the topic list for the given emotion.

    Args:
        emotion (str): The emotion determined from the topic option.
        topic (str): The topic determined from the topic option.
        keyword (str): The keyword to add to the emotion_map.
    """
    print("Checking if we need to update the emotion map.")
    changed = False
    if keyword:
        if emotion in emotion_map and topic not in emotion_map[emotion]:
            emotion_map[emotion].append(topic)
            print(f"Added new topic '{topic}' to emotion '{emotion}'")
            changed = True
    print("Updated emotion map.")
    if changed:
        try:
            save_emotion_map(EMOTION_MAP_PATH, emotion_map)
        except OSError as exc:
            print(
                "Error: could not save emotion map to "
                f"{EMOTION_MAP_PATH} ({exc})."
            )

def get_emotion(topic_option):
    """Return the emotion associated with the topic option, or 'indifference' if not found."""
    for emotion, topics in emotion_map.items():
        if topic_option in topics:
            return emotion
    return "indifference"

def prompt():
    while True:
        original_prompt = input("> ").strip()

        if original_prompt.lower() == "quit":
            print("Goodbye!")
            break

        if not original_prompt:
            print("Please enter something!")
            continue

        # Step 1: Get the topic response from Ollama
        topic_response = topic_keyword_query_ollama(original_prompt)

        if not topic_response:
            print("Could not determine the topic. Is Ollama running?")
            continue

        # Step 2: Get the emotion based on the topic option and/or add keyword to emotion map
        topic = topic_response["topic"]
        emotion = get_emotion(topic)

        update_emotion_map_thread = threading.Thread(target=update_emotion_map, args=(emotion, topic, topic_response["keyword"]))
        update_emotion_map_thread.start()

        # Step 3: Get the final response from Ollama using the new prompt
        final_response = emotion_encoding(emotion, original_prompt)

        if final_response:
            print(f"Topic: {topic}")
            print(f"Emotion: {emotion}")
            print(f"Response: {final_response}")
        else:
            print("Could not generate a response.")
        update_emotion_map_thread.join()

def main():
    print("Emotion Prompt System")
    print("Type your prompt (or 'quit' to exit)")

    prompt()

if __name__ == "__main__":
    main()