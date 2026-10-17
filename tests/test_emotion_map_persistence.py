import argparse
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

emotion_prompt = importlib.import_module("emotion-prompt")


def _load_fresh(argv):
    """Execute emotion-prompt.py with argv installed as sys.argv."""
    module_name = f"emotion_prompt_fresh_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / "emotion-prompt.py"
    )
    module = importlib.util.module_from_spec(spec)
    saved_argv = sys.argv
    sys.argv = argv
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        sys.argv = saved_argv
    return module


def test_builtin_copies_are_independent():
    """Mutating one built-in copy must not affect the next copy."""
    first = emotion_prompt.builtin_emotion_map()
    first["indifference"].append("x")
    second = emotion_prompt.builtin_emotion_map()
    assert second["indifference"] == []
    assert "love" in second["happiness"]


def test_missing_file_uses_builtin_quietly(tmp_path, capsys):
    """A missing file loads the built-in map and is not created."""
    path = tmp_path / "missing.json"
    loaded = emotion_prompt.load_emotion_map(path)
    assert loaded == emotion_prompt.builtin_emotion_map()
    assert capsys.readouterr().out == ""
    assert not path.exists()


@pytest.mark.parametrize("payload", [
    "{not json",
    "",
    "null",
    "[]",
    '"happiness"',
    '{"happiness": "love"}',
    '{"happiness": [1]}',
    '{"happiness": [null]}',
    '{"happiness": {"topic": "love"}}',
    '{"": ["love"]}',
    '{"": []}',
])
def test_corrupt_or_invalid_file_falls_back(tmp_path, capsys, payload):
    """Invalid files warn, fall back, and stay untouched on disk."""
    path = tmp_path / "emotion_map.json"
    path.write_text(payload, encoding="utf-8")
    loaded = emotion_prompt.load_emotion_map(path)
    assert loaded == emotion_prompt.builtin_emotion_map()
    assert path.read_text(encoding="utf-8") == payload
    captured = capsys.readouterr()
    assert "Warning:" in captured.out
    assert "built-in emotion map" in captured.out
    assert str(path) in captured.out


def test_unreadable_directory_falls_back(tmp_path, capsys):
    """A directory where the map should be is treated as unreadable."""
    path = tmp_path / "emotion_map.json"
    path.mkdir()
    loaded = emotion_prompt.load_emotion_map(path)
    assert loaded == emotion_prompt.builtin_emotion_map()
    assert path.is_dir()
    assert "unreadable" in capsys.readouterr().out


def test_saved_map_replaces_builtin_instead_of_merging(tmp_path):
    """A valid file is the whole map. Missing emotions are not filled in."""
    path = tmp_path / "emotion_map.json"
    stored = {"joy": ["sunlight"], "indifference": ["void"]}
    path.write_text(json.dumps(stored), encoding="utf-8")
    loaded = emotion_prompt.load_emotion_map(path)
    assert loaded == stored
    assert "happiness" not in loaded


def test_roundtrip_preserves_topics_and_unicode(tmp_path):
    """Save then load returns the same map, including non-ASCII topics."""
    path = tmp_path / "nested" / "emotion_map.json"
    mapping = emotion_prompt.builtin_emotion_map()
    mapping["indifference"].append("quarks")
    mapping["happiness"].append("café")
    emotion_prompt.save_emotion_map(path, mapping)
    assert path.is_file()
    leftovers = [
        entry for entry in path.parent.iterdir()
        if entry.suffix == ".tmp"
    ]
    assert leftovers == []
    loaded = emotion_prompt.load_emotion_map(path)
    assert loaded == mapping
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "café" in text


def test_save_replaces_atomically_and_keeps_previous_on_failure(tmp_path):
    """A failed rename leaves the previous file and no temp file behind."""
    path = tmp_path / "emotion_map.json"
    original = {"indifference": ["kept"]}
    emotion_prompt.save_emotion_map(path, original)
    updated = {"indifference": ["replaced"]}
    with patch("os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            emotion_prompt.save_emotion_map(path, updated)
    assert json.loads(path.read_text(encoding="utf-8")) == original
    leftovers = [
        entry for entry in tmp_path.iterdir() if entry.suffix == ".tmp"
    ]
    assert leftovers == []


def test_save_renames_a_temp_file_in_the_same_directory(tmp_path):
    """The rename source is a sibling temp file, not the destination."""
    path = tmp_path / "emotion_map.json"
    mapping = {"fear": ["storm"]}
    with patch("os.replace", wraps=os.replace) as mock_replace:
        emotion_prompt.save_emotion_map(path, mapping)
    mock_replace.assert_called_once()
    source, dest = mock_replace.call_args[0]
    assert Path(dest) == path
    assert Path(source).parent == path.parent
    assert Path(source) != path
    assert not Path(source).exists()
    assert json.loads(path.read_text(encoding="utf-8")) == mapping


def test_resolve_path_precedence(tmp_path, monkeypatch):
    """CLI beats env, env beats the default under the home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    env_path = tmp_path / "from-env.json"
    cli_path = tmp_path / "from-cli.json"
    default = emotion_prompt.default_emotion_map_path()
    assert default == tmp_path / ".emotional-prompt" / "emotion_map.json"

    assert emotion_prompt.resolve_emotion_map_path(None, {}) == default
    assert emotion_prompt.resolve_emotion_map_path(
        None, {"EMOTION_MAP_PATH": ""}
    ) == default
    assert emotion_prompt.resolve_emotion_map_path(
        None, {"EMOTION_MAP_PATH": "   "}
    ) == default
    assert emotion_prompt.resolve_emotion_map_path(
        None, {"EMOTION_MAP_PATH": str(env_path)}
    ) == env_path
    assert emotion_prompt.resolve_emotion_map_path(
        str(cli_path), {"EMOTION_MAP_PATH": str(env_path)}
    ) == cli_path
    assert emotion_prompt.resolve_emotion_map_path(
        "~/custom.json", {}
    ) == tmp_path / "custom.json"


def test_cli_option_overrides_env_and_strips_argv(tmp_path):
    """--emotion-map wins over EMOTION_MAP_PATH and is removed from argv."""
    env_path = tmp_path / "env.json"
    cli_path = tmp_path / "cli.json"
    env_path.write_text(
        json.dumps({"happiness": ["fromenv"]}), encoding="utf-8"
    )
    cli_path.write_text(
        json.dumps({"fear": ["fromcli"]}), encoding="utf-8"
    )
    argv = [
        "emotion-prompt.py",
        "--model",
        "mistral",
        "--emotion-map",
        str(cli_path),
        "--port",
        "11435",
    ]
    env = {"EMOTION_MAP_PATH": str(env_path)}
    path, mapping = emotion_prompt.initialize_emotion_map(argv, env)
    assert path == cli_path
    assert mapping == {"fear": ["fromcli"]}
    assert argv == [
        "emotion-prompt.py",
        "--model",
        "mistral",
        "--port",
        "11435",
    ]


def test_cli_equals_form_and_last_flag_wins(tmp_path):
    """--emotion-map=PATH is accepted. The last flag wins."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"joy": ["one"]}), encoding="utf-8")
    second.write_text(json.dumps({"joy": ["two"]}), encoding="utf-8")
    argv = [
        "prog",
        "--emotion-map",
        str(first),
        f"--emotion-map={second}",
        "-m",
        "x",
    ]
    path, mapping = emotion_prompt.initialize_emotion_map(argv, env={})
    assert path == second
    assert mapping == {"joy": ["two"]}
    assert argv == ["prog", "-m", "x"]


def test_cli_flag_requires_a_path():
    """A missing or flag-like value exits and leaves argv alone."""
    missing = ["prog", "--emotion-map"]
    with pytest.raises(SystemExit) as missing_exit:
        emotion_prompt.pop_emotion_map_option(missing)
    assert missing_exit.value.code == 2
    assert missing == ["prog", "--emotion-map"]

    looks_like_flag = ["prog", "--emotion-map", "--model", "mistral"]
    with pytest.raises(SystemExit) as flag_exit:
        emotion_prompt.pop_emotion_map_option(looks_like_flag)
    assert flag_exit.value.code == 2
    assert looks_like_flag == ["prog", "--emotion-map", "--model", "mistral"]

    empty = ["prog", "--emotion-map="]
    with pytest.raises(SystemExit) as empty_exit:
        emotion_prompt.pop_emotion_map_option(empty)
    assert empty_exit.value.code == 2
    assert empty == ["prog", "--emotion-map="]


def test_import_loads_cli_map_and_leaves_other_args(tmp_path, monkeypatch):
    """Startup loads the CLI map and keeps --model/--port for argparse."""
    path = tmp_path / "emotion_map.json"
    stored = emotion_prompt.builtin_emotion_map()
    stored["surprise"].append("comet")
    path.write_text(json.dumps(stored), encoding="utf-8")
    ignored = tmp_path / "ignored.json"
    ignored.write_text(json.dumps({"joy": ["nope"]}), encoding="utf-8")
    monkeypatch.setenv("EMOTION_MAP_PATH", str(ignored))
    argv = [
        "emotion-prompt.py",
        "--port",
        "11435",
        "--emotion-map",
        str(path),
        "--model",
        "mistral",
    ]
    module = _load_fresh(argv)
    assert module.EMOTION_MAP_PATH == path
    assert module.emotion_map["surprise"] == ["gift", "comet"]
    assert argv == [
        "emotion-prompt.py",
        "--port",
        "11435",
        "--model",
        "mistral",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--port", type=int)
    parsed = parser.parse_args(argv[1:])
    assert parsed.model == "mistral"
    assert parsed.port == 11435


def test_import_uses_env_when_flag_absent(tmp_path, monkeypatch):
    """EMOTION_MAP_PATH is used on startup when no CLI option is present."""
    path = tmp_path / "from-env.json"
    path.write_text(
        json.dumps({"indifference": ["tide"]}), encoding="utf-8"
    )
    monkeypatch.setenv("EMOTION_MAP_PATH", str(path))
    argv = ["emotion-prompt.py", "--model", "mistral"]
    module = _load_fresh(argv)
    assert module.EMOTION_MAP_PATH == path
    assert module.emotion_map == {"indifference": ["tide"]}
    assert argv == ["emotion-prompt.py", "--model", "mistral"]


def test_import_uses_default_path_when_unset(tmp_path, monkeypatch):
    """With no flag and no env var, startup uses the home default."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("EMOTION_MAP_PATH", raising=False)
    argv = ["emotion-prompt.py"]
    module = _load_fresh(argv)
    expected = tmp_path / ".emotional-prompt" / "emotion_map.json"
    assert module.EMOTION_MAP_PATH == expected
    assert module.emotion_map == emotion_prompt.builtin_emotion_map()
    assert not expected.exists()
    assert argv == ["emotion-prompt.py"]


def test_update_persists_new_topic_only(tmp_path):
    """A new topic is appended and written. Repeats do not write again."""
    path = tmp_path / "emotion_map.json"
    emotion_prompt.EMOTION_MAP_PATH = path
    emotion_prompt.update_emotion_map(
        "indifference", "cryptography", "ciphers"
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["indifference"] == ["cryptography"]
    assert loaded["happiness"] == ["love", "peace", "food"]
    first_mtime = path.stat().st_mtime_ns

    emotion_prompt.update_emotion_map(
        "indifference", "cryptography", "ciphers"
    )
    assert path.stat().st_mtime_ns == first_mtime
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["indifference"].count("cryptography") == 1
    assert emotion_prompt.emotion_map["indifference"].count(
        "cryptography"
    ) == 1


def test_update_skips_save_without_a_real_change(tmp_path):
    """No keyword, an unknown emotion, or a known topic does not write."""
    path = tmp_path / "emotion_map.json"
    emotion_prompt.EMOTION_MAP_PATH = path
    emotion_prompt.update_emotion_map("indifference", "ghost", None)
    emotion_prompt.update_emotion_map("indifference", "ghost", "")
    emotion_prompt.update_emotion_map("not-an-emotion", "topic", "keyword")
    emotion_prompt.update_emotion_map("happiness", "peace", "calm")
    assert not path.exists()


def test_update_keeps_topic_when_save_fails(tmp_path, capsys):
    """A disk error is reported and the in-memory addition is kept."""
    blocked = tmp_path / "emotion_map.json"
    blocked.mkdir()
    emotion_prompt.EMOTION_MAP_PATH = blocked
    emotion_prompt.update_emotion_map("indifference", "ghosts", "boo")
    assert "ghosts" in emotion_prompt.emotion_map["indifference"]
    captured = capsys.readouterr()
    assert "could not save emotion map" in captured.out
    assert str(blocked) in captured.out


def test_second_run_sees_topic_saved_by_the_first(tmp_path):
    """A later startup loads the topic written by an earlier update."""
    path = tmp_path / "emotion_map.json"
    argv = ["prog", "--emotion-map", str(path)]
    resolved, mapping = emotion_prompt.initialize_emotion_map(argv, env={})
    assert "nebula" not in mapping["indifference"]
    mapping["indifference"].append("nebula")
    emotion_prompt.save_emotion_map(resolved, mapping)

    argv_again = ["prog", "--emotion-map", str(path)]
    resolved_again, mapping_again = emotion_prompt.initialize_emotion_map(
        argv_again, env={}
    )
    assert resolved_again == path
    assert "nebula" in mapping_again["indifference"]
    assert mapping_again["happiness"] == ["love", "peace", "food"]


def test_prompt_persists_new_topic(tmp_path):
    """The interactive loop writes a newly classified topic."""
    path = tmp_path / "emotion_map.json"
    emotion_prompt.EMOTION_MAP_PATH = path
    with patch.object(
        emotion_prompt,
        "query_ollama",
        side_effect=["astronomy telescope", "The sky is wide."],
    ), patch("builtins.input", side_effect=["I like astronomy", "quit"]):
        emotion_prompt.prompt()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert "astronomy" in loaded["indifference"]


def test_prompt_known_topic_and_outage_do_not_write(tmp_path):
    """Known topics and a down Ollama do not create a map file."""
    path = tmp_path / "emotion_map.json"
    emotion_prompt.EMOTION_MAP_PATH = path
    with patch.object(
        emotion_prompt,
        "query_ollama",
        side_effect=["peace harmony", "A calm reply."],
    ), patch("builtins.input", side_effect=["give me peace", "quit"]):
        emotion_prompt.prompt()
    assert not path.exists()

    with patch.object(
        emotion_prompt, "query_ollama", return_value=None
    ), patch("builtins.input", side_effect=["hello", "quit"]):
        emotion_prompt.prompt()
    assert not path.exists()
