"""The `wpt-dataset` commands that read and write data files, thin over the
service layer as wpt-web's HTTP API is: a new empty source, its files as
patches, a conversation, a recipe, with the check's rules and versions."""

import json

from wpt_dataset.cli import main


def run(capsys, *args):
    """(exit code, standard output, the version printed on standard error)."""
    code = main(list(args))
    out, err = capsys.readouterr()
    versions = [line.removeprefix("version: ") for line in err.splitlines() if line.startswith("version: ")]
    return code, out, versions[0] if versions else None


def test_the_commands_write_a_new_source_its_files_and_a_conversation(kit, repo, store, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    root = ["--root", str(repo.root)]
    (tmp_path / "license.txt").write_text("CC0 1.0 Universal\n")
    (tmp_path / "poem.txt").write_text("the night falls on the town\n")
    code, out, _ = run(capsys, *root, "new-source", "mine", "poems", "--license", "CC0-1.0", "--attribution",
                       "A. Person", "--license-text", str(tmp_path / "license.txt"), "--by", "A. Person",
                       "--file", f"poem.txt={tmp_path / 'poem.txt'}", "--chat", "sft/talks.jsonl")
    sid = out.strip()
    assert code == 0 and sid.startswith("mine-")

    code, out, version = run(capsys, *root, "files", sid)
    assert code == 0 and out == "100644        28 text [] poem.txt\n"
    (tmp_path / "poem2.txt").write_text("the night falls on the quiet town\n")
    assert run(capsys, *root, "write", sid, "poem.txt", str(tmp_path / "poem2.txt"), "--patch", "quiet",
               "--version", version)[0] == 0
    assert run(capsys, *root, "write", sid, "poem.txt", str(tmp_path / "poem.txt"), "--patch", "late",
               "--version", version)[0] == 1  # stale: the file changed since that version
    code, out, version = run(capsys, *root, "read", sid, "poem.txt")
    assert out == "the night falls on the quiet town\n"

    conversation = {"schema": 3, "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                                              {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}]}
    (tmp_path / "talk.json").write_text(json.dumps(conversation))
    assert run(capsys, *root, "save-conversation", sid, "sft/talks.jsonl", "new", str(tmp_path / "talk.json"),
               "--patch", "talk", "--version", version)[0] == 0
    code, out, version = run(capsys, *root, "conversations", sid, "sft/talks.jsonl")
    assert out == ("line 1: 2 messages (user, assistant); 0 tools; 0 model drafts, 0 of them changed; "
                   "metadata: none\n")
    assert [p.name for p in repo.source(sid).patches] == ["0001-texts.patch", "0002-quiet.patch", "0003-talk.patch"]

    (tmp_path / "half.json").write_text(json.dumps({"schema": 1, "pools": [{"name": "a", "select": "a",
                                                                            "weight": 0.5}]}))
    assert run(capsys, *root, "save", "mixture", "half", str(tmp_path / "half.json"), "--version", "")[0] == 1
    assert run(capsys, *root, "list", "mixture") == (0, "", None)
    assert repo.check() == []


def test_new_source_without_its_texts_leaves_the_license_text_to_a_person(kit, repo, store, tmp_path, monkeypatch,
                                                                         capsys):
    # what an AI assistant may run: new-source without --file and --license-text; the check names the source
    # until a person writes the license text, and files given without that text are refused, leaving nothing
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    root = ["--root", str(repo.root)]
    common = ["--license", "CC0-1.0", "--attribution", "A. Person", "--by", "A. Person"]
    (tmp_path / "poem.txt").write_text("the night falls on the town\n")
    given = f"poem.txt={tmp_path / 'poem.txt'}"
    assert run(capsys, *root, "new-source", "mine", "early", *common, "--file", given)[0] == 1
    assert repo.source_ids() == []

    code, out, _ = run(capsys, *root, "new-source", "mine", "poems", *common, "--chat", "sft/talks.jsonl")
    sid = out.strip()
    assert code == 0 and repo.source_ids() == [sid]
    document = repo.raw("source", sid)
    assert document["licenses"][0]["text"] == "" and document["formats"] == [{"format": "chat",
                                                                               "paths": ["sft/talks.jsonl"]}]
    # a new source is under review until a maintainer decides (maintainer's decision, 2026-10-22)
    assert document["decision"] == {"decision": "under-review", "by": "A. Person", "date": document["revision"]}
    # the check names it as a note and does not fail; readers leave it out with the same note
    code, out, _ = run(capsys, *root, "check", "--fetch", sid)
    assert code == 0 and out.splitlines()[0] == repo.source(sid).note and "entries" not in out
    notes = []
    assert repo.sources(notes.append) == [] and notes == [repo.source(sid).note]

    document["licenses"][0]["text"] = "CC0 1.0 Universal\n"  # the person writes it
    repo.save("source", sid, document, repo.version("source", sid))
    assert repo.check_source(sid) == [] and not repo.source(sid).unwritten
    code, out, _ = run(capsys, *root, "check")
    assert code == 0 and "left out" not in out


def test_new_source_refuses_an_empty_license_text_file_and_declares_formats_without_files(
        kit, repo, store, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    root = ["--root", str(repo.root)]
    common = ["--license", "CC0-1.0", "--attribution", "A. Person", "--by", "A. Person"]
    (tmp_path / "empty.txt").write_bytes(b"")
    code, out, _ = run(capsys, *root, "new-source", "mine", "empty", *common, "--license-text",
                       str(tmp_path / "empty.txt"))
    assert code == 1 and repo.source_ids() == []

    code, out, _ = run(capsys, *root, "new-source", "mine", "formats", *common, "--format", "markdown=**/*.md",
                       "--format", "text=notes/*.txt")
    sid = out.strip()
    assert code == 0 and repo.source(sid).data["formats"] == [{"format": "markdown", "paths": ["**/*.md"]},
                                                              {"format": "text", "paths": ["notes/*.txt"]}]
    assert repo.source_files(sid, store)[1] == {}  # metadata only, no file
    assert run(capsys, *root, "new-source", "mine", "bad", *common, "--format", "markdown")[0] == 1
    document = repo.raw("source", sid)  # a person writes the license text, then the files come in either format
    document["licenses"][0]["text"] = "CC0 1.0 Universal\n"
    version = repo.save("source", sid, document, repo.version("source", sid))
    repo.change_source_files(sid, {"docs/a.md": b"# A\n", "notes/b.txt": b"b\n"}, "files", repo.source_version(sid),
                             store)
    assert sorted(repo.source_files(sid, store)[1]) == ["docs/a.md", "notes/b.txt"]
