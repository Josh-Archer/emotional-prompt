import importlib.util
import json
import os
import subprocess
import sys
import threading
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
    """Invalid files are moved aside and the built-in map is used."""
    path = tmp_path / "emotion_map.json"
    path.write_text(payload, encoding="utf-8")
    loaded = emotion_prompt.load_emotion_map(path)
    assert loaded == emotion_prompt.builtin_emotion_map()
    backups = list(tmp_path.glob("emotion_map.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == payload
    assert not path.exists()
    captured = capsys.readouterr()
    assert "Warning:" in captured.out
    assert "built-in emotion map" in captured.out
    assert str(path) in captured.out
    assert str(backups[0]) in captured.out
    assert "moved it aside" in captured.out


def test_unreadable_directory_falls_back(tmp_path, capsys):
    """A directory where the map should be is treated as unreadable."""
    path = tmp_path / "emotion_map.json"
    path.mkdir()
    loaded = emotion_prompt.load_emotion_map(path)
    assert loaded == emotion_prompt.builtin_emotion_map()
    assert path.is_dir()
    assert "unreadable" in capsys.readouterr().out
    assert emotion_prompt._overwrite_is_blocked(path)


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
    with patch("os.replace", wraps=os.replace) as mock_replace, patch(
        "os.fsync", wraps=os.fsync
    ) as mock_fsync:
        emotion_prompt.save_emotion_map(path, mapping)
    mock_replace.assert_called_once()
    source, dest = mock_replace.call_args[0]
    assert Path(dest) == path
    assert Path(source).parent == path.parent
    assert Path(source) != path
    assert not Path(source).exists()
    assert mock_fsync.call_count == 2
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


def test_cli_option_overrides_env_without_mutating_args(tmp_path, monkeypatch):
    """--emotion-map wins over EMOTION_MAP_PATH without mutating args."""
    env_path = tmp_path / "env.json"
    cli_path = tmp_path / "cli.json"
    env_path.write_text(
        json.dumps({"happiness": ["fromenv"]}), encoding="utf-8"
    )
    cli_path.write_text(
        json.dumps({"fear": ["fromcli"]}), encoding="utf-8"
    )
    monkeypatch.setenv("EMOTION_MAP_PATH", str(env_path))
    args = ["--model", "mistral", "--emotion-map", str(cli_path)]
    with patch("builtins.input", return_value="quit"):
        emotion_prompt.main(args)
    assert emotion_prompt.EMOTION_MAP_PATH == cli_path
    assert emotion_prompt.emotion_map == {"fear": ["fromcli"]}
    assert args == ["--model", "mistral", "--emotion-map", str(cli_path)]


def test_cli_equals_form_and_last_flag_wins(tmp_path):
    """argparse accepts equals form and uses the last option value."""
    second = tmp_path / "second.json"
    parsed = emotion_prompt.parse_args([
        "--emotion-map=first.json",
        f"--emotion-map={second}",
        "-m",
        "x",
    ])
    assert parsed.emotion_map == str(second)
    assert parsed.model == "x"


@pytest.mark.parametrize("args", [
    ["--emotion-map"],
    ["--emotion-map", "--model", "mistral"],
    ["--emotion-map="],
    ["--emotion-map", ""],
    ["--emotion-map", "   "],
])
def test_cli_flag_requires_a_nonempty_path(args, capsys):
    """A missing, flag-like, or empty value exits with a clear error."""
    with pytest.raises(SystemExit) as missing_exit:
        emotion_prompt.parse_args(args)
    assert missing_exit.value.code == 2
    error = capsys.readouterr().err
    assert "--emotion-map" in error
    if args in (["--emotion-map="], ["--emotion-map", ""],
                ["--emotion-map", "   "]):
        assert "non-empty path" in error


def test_import_does_not_read_the_home_or_cli_map(tmp_path, monkeypatch):
    """Import leaves argv alone and does not read a map off disk."""
    home = tmp_path / "home"
    home_map = home / ".emotional-prompt" / "emotion_map.json"
    home_map.parent.mkdir(parents=True)
    home_map.write_text(
        json.dumps({"joy": ["fromhome"]}), encoding="utf-8"
    )
    cli_path = tmp_path / "cli.json"
    stored = emotion_prompt.builtin_emotion_map()
    stored["surprise"].append("comet")
    cli_path.write_text(json.dumps(stored), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("EMOTION_MAP_PATH", str(home_map))
    argv = [
        "emotion-prompt.py",
        "--port",
        "11435",
        "--emotion-map",
        str(cli_path),
        "--model",
        "mistral",
    ]
    module = _load_fresh(argv)
    assert module.EMOTION_MAP_PATH is None
    assert module.emotion_map == emotion_prompt.builtin_emotion_map()
    assert "comet" not in module.emotion_map["surprise"]
    assert argv == [
        "emotion-prompt.py",
        "--port",
        "11435",
        "--emotion-map",
        str(cli_path),
        "--model",
        "mistral",
    ]
    assert json.loads(home_map.read_text(encoding="utf-8")) == {
        "joy": ["fromhome"]
    }
    parsed = module.parse_args(argv[1:])
    assert parsed.model == "mistral"
    assert parsed.port == 11435
    assert parsed.emotion_map == str(cli_path)
    with patch("builtins.input", return_value="quit"):
        module.main(argv[1:])
    assert module.EMOTION_MAP_PATH == cli_path
    assert module.emotion_map["surprise"] == ["gift", "comet"]
    assert module.MODEL == "mistral"
    assert module.OLLAMA_PORT == 11435
    assert json.loads(home_map.read_text(encoding="utf-8")) == {
        "joy": ["fromhome"]
    }


def test_main_uses_env_when_flag_absent(tmp_path, monkeypatch):
    """EMOTION_MAP_PATH is used when main runs with no CLI option."""
    path = tmp_path / "from-env.json"
    path.write_text(
        json.dumps({"indifference": ["tide"]}), encoding="utf-8"
    )
    monkeypatch.setenv("EMOTION_MAP_PATH", str(path))
    argv = ["emotion-prompt.py", "--model", "mistral"]
    module = _load_fresh(argv)
    assert module.emotion_map == emotion_prompt.builtin_emotion_map()
    with patch("builtins.input", return_value="quit"):
        module.main(argv[1:])
    assert module.EMOTION_MAP_PATH == path
    assert module.emotion_map == {"indifference": ["tide"]}
    assert module.MODEL == "mistral"
    assert argv == ["emotion-prompt.py", "--model", "mistral"]


def test_main_uses_default_path_when_unset(tmp_path, monkeypatch):
    """With no flag and no env var, main uses the home default."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("EMOTION_MAP_PATH", raising=False)
    argv = ["emotion-prompt.py"]
    module = _load_fresh(argv)
    expected = tmp_path / ".emotional-prompt" / "emotion_map.json"
    assert module.EMOTION_MAP_PATH is None
    assert module.emotion_map == emotion_prompt.builtin_emotion_map()
    assert not expected.exists()
    with patch("builtins.input", return_value="quit"):
        module.main(argv[1:])
    assert module.EMOTION_MAP_PATH == expected
    assert module.emotion_map == emotion_prompt.builtin_emotion_map()
    assert not expected.exists()
    assert argv == ["emotion-prompt.py"]


def test_help_lists_emotion_map_with_model_and_port(capsys):
    """--emotion-map is registered next to --model and --port."""
    with pytest.raises(SystemExit) as exc_info:
        emotion_prompt.parse_args(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--emotion-map" in captured.out
    assert "--model" in captured.out
    assert "-m" in captured.out
    assert "--port" in captured.out
    assert "-p" in captured.out
    assert "llama3.2" in captured.out
    assert "11434" in captured.out
    combined = emotion_prompt.parse_args(
        ["--model", "mistral", "--port", "11435", "--emotion-map", "x.json"]
    )
    assert combined.model == "mistral"
    assert combined.port == 11435
    assert combined.emotion_map == "x.json"


def test_cli_help_does_not_load_the_home_map(tmp_path):
    """python emotion-prompt.py --help lists --emotion-map and exits."""
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("EMOTION_MAP_PATH", None)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "emotion-prompt.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    assert "--emotion-map" in result.stdout
    assert "--model" in result.stdout
    assert "--port" in result.stdout
    assert not (home / ".emotional-prompt").exists()


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


def test_update_keeps_topic_when_save_fails(tmp_path, caplog):
    """A disk error is reported and the in-memory addition is kept."""
    blocked = tmp_path / "emotion_map.json"
    blocked.mkdir()
    emotion_prompt.EMOTION_MAP_PATH = blocked
    emotion_prompt.update_emotion_map("indifference", "ghosts", "boo")
    assert "ghosts" in emotion_prompt.emotion_map["indifference"]
    assert "Could not update emotion map" in caplog.text
    assert str(blocked) in caplog.text


def test_second_run_sees_topic_saved_by_the_first(tmp_path):
    """A later startup loads the topic written by an earlier update."""
    path = tmp_path / "emotion_map.json"
    emotion_prompt.EMOTION_MAP_PATH = path
    emotion_prompt.update_emotion_map("indifference", "nebula", "nebula")
    emotion_prompt.EMOTION_MAP_PATH = path
    emotion_prompt._emotion_map_source = None
    emotion_prompt.emotion_map.clear()
    emotion_prompt.emotion_map.update(emotion_prompt.builtin_emotion_map())
    emotion_prompt.ensure_emotion_map_loaded()
    assert "nebula" in emotion_prompt.emotion_map["indifference"]
    assert emotion_prompt.emotion_map["happiness"] == ["love", "peace", "food"]


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


def test_update_backs_up_corrupt_map_before_save(tmp_path, capsys):
    """Learning a topic must not destroy a corrupt map file."""
    path = tmp_path / "emotion_map.json"
    original = "{not json"
    path.write_text(original, encoding="utf-8")
    emotion_prompt.EMOTION_MAP_PATH = path
    emotion_prompt._emotion_map_source = None
    emotion_prompt.update_emotion_map("indifference", "dogs", "dogs")
    backups = sorted(tmp_path.glob("emotion_map.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == original
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "dogs" in saved["indifference"]
    assert saved["happiness"] == ["love", "peace", "food"]
    captured = capsys.readouterr()
    assert "moved it aside" in captured.out
    assert str(backups[0]) in captured.out
    assert backups[0].read_text(encoding="utf-8") == original


def test_unreadable_map_cannot_be_overwritten_by_learning(
    tmp_path, capsys, caplog
):
    """A read failure blocks saves before the first learned topic."""
    path = tmp_path / "emotion_map.json"
    original = '{"love": ["precious", "user", "data"]}\n'
    path.write_text(original, encoding="utf-8")
    emotion_prompt.EMOTION_MAP_PATH = path
    with patch.object(
        Path, "read_text", side_effect=PermissionError("denied")
    ):
        emotion_prompt.update_emotion_map("indifference", "dogs", "dogs")
    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("emotion_map.json.corrupt-*"))
    assert emotion_prompt._overwrite_is_blocked(path)
    output = capsys.readouterr().out
    assert "Refusing to overwrite it" in output
    assert "could not be read or backed up" in caplog.text
    assert "corrupt copy" not in caplog.text


def test_non_utf8_map_is_backed_up_and_get_emotion_falls_back(tmp_path):
    """Invalid UTF-8 is quarantined and does not crash get_emotion."""
    path = tmp_path / "emotion_map.json"
    original = b'{"love":["user-data"]}\xff\xfe'
    path.write_bytes(original)
    emotion_prompt.EMOTION_MAP_PATH = path
    emotion_prompt._emotion_map_source = None
    assert emotion_prompt.get_emotion("user-data") == "indifference"
    backups = list(tmp_path.glob("emotion_map.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert not path.exists()


def test_main_handles_non_utf8_map_without_crashing(tmp_path, monkeypatch):
    """Startup quarantines non-UTF-8 maps and reaches the prompt loop."""
    path = tmp_path / "bad.json"
    original = b"\xff\xfe"
    path.write_bytes(original)
    monkeypatch.delenv("EMOTION_MAP_PATH", raising=False)
    with patch("builtins.input", return_value="quit"):
        emotion_prompt.main(["--emotion-map", str(path)])
    backups = list(tmp_path.glob("bad.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert not path.exists()


def test_corrupt_backups_get_unique_names(tmp_path, monkeypatch):
    """Two corrupt loads in one timestamp do not share a backup path."""
    monkeypatch.setattr(
        emotion_prompt,
        "_corrupt_backup_stamp",
        lambda: "20261017T000000000000",
    )
    path = tmp_path / "emotion_map.json"
    path.write_text("{bad", encoding="utf-8")
    emotion_prompt.load_emotion_map(path)
    path.write_text("{worse", encoding="utf-8")
    emotion_prompt.load_emotion_map(path)
    names = sorted(p.name for p in tmp_path.glob("emotion_map.json.corrupt-*"))
    assert names == [
        "emotion_map.json.corrupt-20261017T000000000000",
        "emotion_map.json.corrupt-20261017T000000000000-1",
    ]
    assert (tmp_path / names[0]).read_text(encoding="utf-8") == "{bad"
    assert (tmp_path / names[1]).read_text(encoding="utf-8") == "{worse"


def test_refuses_to_overwrite_when_corrupt_backup_fails(
    tmp_path, capsys, caplog
):
    """If the corrupt file cannot be moved, a later save leaves it alone."""
    path = tmp_path / "emotion_map.json"
    original = "{not json"
    path.write_text(original, encoding="utf-8")
    with patch("os.replace", side_effect=OSError("permission denied")):
        loaded = emotion_prompt.load_emotion_map(path)
    assert loaded == emotion_prompt.builtin_emotion_map()
    assert path.read_text(encoding="utf-8") == original
    assert "Refusing to overwrite" in capsys.readouterr().out

    emotion_prompt.EMOTION_MAP_PATH = path
    emotion_prompt._emotion_map_source = path
    emotion_prompt.emotion_map.clear()
    emotion_prompt.emotion_map.update(loaded)
    emotion_prompt.update_emotion_map("indifference", "dogs", "dogs")
    assert path.read_text(encoding="utf-8") == original
    assert "backed up" in caplog.text


def test_save_writes_through_a_symlink(tmp_path):
    """Saving replaces the symlink target and keeps the link itself."""
    target = tmp_path / "real.json"
    link = tmp_path / "link.json"
    target.write_text('{"fear": ["old"]}\n', encoding="utf-8")
    link.symlink_to(target.name)
    mapping = {"fear": ["storm"]}
    with patch("os.replace", wraps=os.replace) as mock_replace:
        emotion_prompt.save_emotion_map(link, mapping)
    replaced = Path(mock_replace.call_args[0][1])
    assert replaced == target.resolve()
    assert link.is_symlink()
    assert os.readlink(link) == target.name
    assert json.loads(target.read_text(encoding="utf-8")) == mapping
    assert json.loads(link.read_text(encoding="utf-8")) == mapping


def test_corrupt_symlink_backs_up_the_target(tmp_path, capsys):
    """A corrupt symlink target is moved aside; the link is not replaced."""
    target = tmp_path / "real.json"
    link = tmp_path / "emotion_map.json"
    original = "{not json"
    target.write_text(original, encoding="utf-8")
    link.symlink_to(target.name)
    loaded = emotion_prompt.load_emotion_map(link)
    assert loaded == emotion_prompt.builtin_emotion_map()
    assert link.is_symlink()
    assert os.readlink(link) == target.name
    assert not target.exists()
    backups = list(tmp_path.glob("real.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == original
    assert str(backups[0]) in capsys.readouterr().out

    mapping = {"joy": ["sun"]}
    emotion_prompt.save_emotion_map(link, mapping)
    assert link.is_symlink()
    assert os.readlink(link) == target.name
    assert json.loads(target.read_text(encoding="utf-8")) == mapping
    assert backups[0].read_text(encoding="utf-8") == original


def test_lazy_load_uses_the_redirected_path(tmp_path, monkeypatch):
    """A lazy load reads the fixture path, not the home map."""
    home = tmp_path / "home"
    home_map = home / ".emotional-prompt" / "emotion_map.json"
    home_map.parent.mkdir(parents=True)
    home_map.write_text(
        json.dumps({"joy": ["fromhome"]}), encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("EMOTION_MAP_PATH", raising=False)
    path = emotion_prompt.EMOTION_MAP_PATH
    path.write_text(
        json.dumps({"joy": ["sunlight"]}), encoding="utf-8"
    )
    emotion_prompt._emotion_map_source = None
    loaded_path = emotion_prompt.ensure_emotion_map_loaded()
    assert loaded_path == path
    assert emotion_prompt.emotion_map == {"joy": ["sunlight"]}
    assert json.loads(home_map.read_text(encoding="utf-8")) == {
        "joy": ["fromhome"]
    }


def test_update_thread_logs_runtime_error(tmp_path, caplog):
    """An unexpected save error is logged and does not kill the thread."""
    path = tmp_path / "emotion_map.json"
    emotion_prompt.EMOTION_MAP_PATH = path
    emotion_prompt._emotion_map_source = path
    seen = []

    def hook(args):
        seen.append(args)

    with patch.object(
        emotion_prompt,
        "save_emotion_map",
        side_effect=RuntimeError("dict changed size"),
    ), patch("threading.excepthook", hook):
        worker = threading.Thread(
            target=emotion_prompt.update_emotion_map,
            args=("indifference", "ghosts", "boo"),
        )
        worker.start()
        worker.join()
    assert seen == []
    assert not worker.is_alive()
    assert "ghosts" in emotion_prompt.emotion_map["indifference"]
    assert "Could not update emotion map" in caplog.text
    assert "dict changed size" in caplog.text
    assert not path.exists()


def test_update_thread_logs_failures_before_saving(tmp_path, caplog):
    """Unexpected failures anywhere in the worker do not escape the thread."""
    emotion_prompt.EMOTION_MAP_PATH = tmp_path / "emotion_map.json"
    seen = []

    def hook(args):
        seen.append(args)

    with patch.object(
        emotion_prompt,
        "ensure_emotion_map_loaded",
        side_effect=RuntimeError("map load race"),
    ), patch("threading.excepthook", hook):
        worker = threading.Thread(
            target=emotion_prompt.update_emotion_map,
            args=("indifference", "ghosts", "boo"),
        )
        worker.start()
        worker.join()

    assert seen == []
    assert not worker.is_alive()
    assert "Could not update emotion map" in caplog.text
    assert "map load race" in caplog.text
    assert not emotion_prompt.EMOTION_MAP_PATH.exists()
