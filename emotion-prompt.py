import datetime
import os
import tempfile
from pathlib import Path
import logging

try:
    import requests
except ImportError:
    print("Error:  module not found. Install it with 'pip install requests'.")
    exit(1)

import argparse
import json
import utils
import threading

# Define the model to use
MODEL = "llama3.2"
logger = logging.getLogger(__name__)

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


def _destination_path(path):
    """Return the file to replace.

    ``os.replace`` on a symlink replaces the link with a regular file.
    Follow the link so writes update the target and leave the link in place.
    """
    path = Path(path)
    if path.is_symlink():
        return path.resolve()
    return path


def _corrupt_backup_stamp():
    """Timestamp used in ``<name>.corrupt-<stamp>`` backup names."""
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")


def _unique_corrupt_backup(source):
    """Return a sibling path that does not exist yet."""
    source = Path(source)
    stamp = _corrupt_backup_stamp()
    candidate = source.with_name(f"{source.name}.corrupt-{stamp}")
    counter = 1
    while candidate.exists() or candidate.is_symlink():
        candidate = source.with_name(
            f"{source.name}.corrupt-{stamp}-{counter}"
        )
        counter += 1
    return candidate


def _block_tokens(path):
    path = Path(path)
    tokens = {str(path)}
    try:
        tokens.add(str(path.resolve()))
    except OSError:
        pass
    return tokens


def _block_overwrite(path):
    _overwrite_blocked.update(_block_tokens(path))
    followed = _destination_path(path)
    _overwrite_blocked.update(_block_tokens(followed))


def _allow_overwrite(path):
    _overwrite_blocked.difference_update(_block_tokens(path))
    try:
        followed = _destination_path(path)
    except OSError:
        return
    _overwrite_blocked.difference_update(_block_tokens(followed))


def _overwrite_is_blocked(path):
    if _block_tokens(path) & _overwrite_blocked:
        return True
    try:
        followed = _destination_path(path)
    except OSError:
        return False
    return bool(_block_tokens(followed) & _overwrite_blocked)


def _quarantine_corrupt_map(path, reason):
    """Move a corrupt map aside so a later save cannot destroy it.

    Returns the backup path. If the file cannot be moved, remember the
    path and refuse any later overwrite. The original bytes stay put.
    """
    source = _destination_path(path)
    backup = _unique_corrupt_backup(source)
    try:
        os.replace(source, backup)
    except OSError as exc:
        _block_overwrite(path)
        _warn_emotion_map(
            path,
            f"{reason}; could not move it aside ({exc}). "
            "Refusing to overwrite it",
        )
        return None
    _allow_overwrite(path)
    _warn_emotion_map(path, f"{reason}; moved it aside to {backup}")
    return backup


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
    unreadable file yields the built-in map, with a warning, and is left
    in place. Invalid JSON or invalid contents are moved aside to
    ``<filename>.corrupt-<timestamp>`` before any later save, and the
    warning names that backup. The built-in map is returned. If the
    corrupt file cannot be moved, it is left in place and later saves
    refuse to overwrite it.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        _allow_overwrite(path)
        return builtin_emotion_map()
    except OSError as exc:
        _block_overwrite(path)
        _warn_emotion_map(
            path,
            f"unreadable: {exc}. Refusing to overwrite it until it can be "
            "read or removed",
        )
        return builtin_emotion_map()
    except UnicodeDecodeError as exc:
        _quarantine_corrupt_map(path, f"invalid UTF-8: {exc}")
        return builtin_emotion_map()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _quarantine_corrupt_map(path, f"invalid JSON: {exc}")
        return builtin_emotion_map()

    if not _is_valid_emotion_map(data):
        _quarantine_corrupt_map(path, "invalid contents")
        return builtin_emotion_map()
    _allow_overwrite(path)
    return {key: list(topics) for key, topics in data.items()}


def save_emotion_map(path, mapping):
    """Atomically replace the map file with the JSON encoding of mapping.

    A temp file in the same directory is written and fsynced, then renamed
    into place. A crash before the rename leaves the previous file intact.
    If path is a symlink, the target file is replaced and the link stays.
    A path whose corrupt contents could not be backed up is not replaced.
    """
    requested = Path(path)
    path = _destination_path(requested)
    if _overwrite_is_blocked(requested) or _overwrite_is_blocked(path):
        raise OSError(
            "refusing to overwrite emotion map at "
            f"{requested} because it could not be read or backed up"
        )
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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def active_emotion_map_path():
    """Return the map path, choosing the default only on first use."""
    global EMOTION_MAP_PATH
    if EMOTION_MAP_PATH is None:
        EMOTION_MAP_PATH = resolve_emotion_map_path(None)
    return Path(EMOTION_MAP_PATH)


def ensure_emotion_map_loaded():
    """Load ``EMOTION_MAP_PATH`` into ``emotion_map`` once per path.

    Import does not call this, so a test can redirect ``EMOTION_MAP_PATH``
    before the real home map is read.
    """
    global _emotion_map_source
    path = active_emotion_map_path()
    if _emotion_map_source == path:
        return path
    loaded = load_emotion_map(path)
    emotion_map.clear()
    emotion_map.update(loaded)
    _emotion_map_source = path
    return path


# Not loaded from disk at import. main() loads after argparse, and the
# first get/update loads after tests redirect EMOTION_MAP_PATH.
EMOTION_MAP_PATH = None
emotion_map = builtin_emotion_map()
_emotion_map_source = None
_overwrite_blocked = set()

OLLAMA_PORT = 11434


def topic_keyword_query_ollama(input_prompt):
    """
    Query Ollama with a prompt.
    """

    ollama_prompt = (
        f"Respond with a single topic and the keyword from the prompt "
        "that made you pick it. "
        "Be concise, do not give any extra context other than the 2 words. "
        "If you respond with more or less than that we will break. "
        "The first word is the topic and the second is the keyword in the "
        "prompt that made you pick it. "
        "The keyword has to be a word that is in the prompt. "
        f"Here is the prompt: {input_prompt}"
    )

    result = query_ollama(ollama_prompt)
    if result is None:
        return None
    print(result)
    response_dict = utils.split_to_dict(result.lower())
    response_dict["keyword"] = (
        response_dict["keyword"] if "keyword" in response_dict else None
    )
    keyword = response_dict["keyword"] or ""
    print(f'{response_dict["topic"]}: {keyword}')
    return response_dict


def emotion_encoding(emotion, input_prompt):
    input_prompt = (
        f"{input_prompt}. Respond with you having the Emotion of {emotion}. "
        "But don't let on that you are being told to respond with this "
        "emotion."
    )
    return query_ollama(input_prompt)


def query_ollama(input_prompt):
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/generate"

    payload = {
        "model": MODEL,
        "prompt": input_prompt,
        "stream": False
    }

    try:
        print(f"Prompt: {input_prompt}")
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        result = json.loads(response.text)
        print(f'Response: {result["response"]}')
        return result['response']
    except requests.exceptions.ConnectionError:
        print(
            f"Error: Could not connect to Ollama on 127.0.0.1:{OLLAMA_PORT}. "
            "Is it running?"
        )
        print(f"Run 'ollama run {MODEL}' in a terminal and keep it open.")
        return None
    except requests.exceptions.RequestException as exc:
        print(f"Error connecting to Ollama: {exc}")
        return None


def extract_topic_option(topic_response):
    """Extract the first word from the topic response as the topic option."""
    if not topic_response:
        return "indifference"
    words = topic_response.split()
    return words[0] if words else "indifference"


def update_emotion_map(emotion, topic, keyword):
    """
    Updates the emotion_map by adding a topic to the matching emotion.

    Args:
        emotion (str): The emotion determined from the topic option.
        topic (str): The topic determined from the topic option.
        keyword (str): The keyword to add to the emotion_map.
    """
    try:
        ensure_emotion_map_loaded()
        print("Checking if we need to update the emotion map.")
        changed = False
        if (
            keyword
            and emotion in emotion_map
            and topic not in emotion_map[emotion]
        ):
            emotion_map[emotion].append(topic)
            print(f"Added new topic '{topic}' to emotion '{emotion}'")
            changed = True
        print("Updated emotion map.")
        if changed:
            path = active_emotion_map_path()
            save_emotion_map(path, emotion_map)
    except Exception:
        logger.exception("Could not update emotion map")


def get_emotion(topic_option):
    """Return the topic's emotion, or 'indifference' if it is not found."""
    ensure_emotion_map_loaded()
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

        # Step 2: Find the emotion and add the topic to the map if needed.
        topic = topic_response["topic"]
        emotion = get_emotion(topic)

        update_emotion_map_thread = threading.Thread(
            target=update_emotion_map,
            args=(emotion, topic, topic_response["keyword"]),
        )
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


def parse_args(args=None):
    """Parse command-line arguments for model, port, and the emotion map.

    ``--model`` and ``--port`` are the same options used by the model/port
    CLI. ``--emotion-map`` selects the on-disk map and is listed in
    ``--help``.
    """
    parser = argparse.ArgumentParser(
        description="Emotion Prompt System"
    )

    def nonempty_map_path(value):
        if value.strip() == "":
            raise argparse.ArgumentTypeError(
                "--emotion-map requires a non-empty path"
            )
        return value

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=MODEL,
        help=f"Ollama model to use (default: {MODEL})",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=OLLAMA_PORT,
        help=f"Ollama server port (default: {OLLAMA_PORT})",
    )
    parser.add_argument(
        "--emotion-map",
        default=None,
        type=nonempty_map_path,
        metavar="PATH",
        help=(
            "JSON file for the emotion map "
            "(default: $EMOTION_MAP_PATH or "
            "~/.emotional-prompt/emotion_map.json)"
        ),
    )
    return parser.parse_args(args)


def main(args=None):
    """Load configuration and run the interactive prompt."""
    global MODEL, OLLAMA_PORT, EMOTION_MAP_PATH, _emotion_map_source
    parsed = parse_args(args)
    MODEL = parsed.model
    OLLAMA_PORT = parsed.port
    EMOTION_MAP_PATH = resolve_emotion_map_path(parsed.emotion_map)
    _emotion_map_source = None
    ensure_emotion_map_loaded()

    print("Emotion Prompt System")
    print("Type your prompt (or 'quit' to exit)")

    prompt()


if __name__ == "__main__":
    main()
