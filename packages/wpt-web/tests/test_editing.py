"""Editing every kind of data through wpt-web's HTTP API: a person's own
text in a new empty source, conversations with model drafts, tasks,
recipes, Dockerfiles and signer lists, all through the service layer and
its rules, with git's status shown and nothing committed."""

import http.client
import json
import subprocess
import threading
from datetime import date
from urllib.parse import quote

import pytest

from wpt_dataset.repo import Repo
from wpt_store import open_store
from wpt_web import pages
from wpt_web.drafts import DraftFailed, Drafter, MachineFault, Unloadable, UnknownFault
from wpt_web.server import make_server

LICENSE_TEXT = "I give this text to the public domain, CC0 1.0 Universal.\n"
TEXT = "A first line of my own.\n\nA second paragraph.\n"


@pytest.fixture
def web(tmp_path):
    """A git repository with no source yet, served on a free port with its
    own store; gives request(method, target, body) and the repository."""
    root = tmp_path / "wpt"
    (root / "data" / "sources").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    store = open_store((tmp_path / "store").as_uri())
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<p>interface</p>")
    server = make_server(pages.routes(Repo(root), store, Drafter()), static)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    def request(method, target, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", port)
        connection.request(method, target, None if body is None else json.dumps(body).encode(),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    yield request, Repo(root), store
    server.shutdown()
    server.server_close()


def new_source(request, files, name="notes"):
    status, made = request("POST", "/api/sources", {
        "origin": "mine", "item": name, "name": name, "license": "CC0-1.0", "attribution": "A. Person, notes",
        "license_text": LICENSE_TEXT, "by": "A. Person", "tags": ["en"], "files": files, "patch": "first-texts"})
    assert status == 200, made
    return made["id"]


def path(text):
    return quote(text, safe="")


def test_a_persons_text_starts_an_empty_source_and_every_change_is_a_patch(web):
    # "New source": the text becomes the first patch of an empty source under
    # review; an edit becomes the next patch; a save from an old version is refused
    request, repo, store = web
    sid = new_source(request, {"notes/one.md": {"format": "markdown", "text": TEXT}})
    assert [p.name for p in repo.source(sid).patches] == ["0001-first-texts.patch"]
    assert repo.source(sid).data["decision"] == {"decision": "under-review", "by": "A. Person",
                                                 "date": date.today().isoformat()}
    status, listing = request("GET", f"/api/sources/{sid}/files")
    assert status == 200 and listing["files"] == {"notes/one.md": {"mode": "100644", "size": len(TEXT),
                                                                  "format": "markdown", "tags": ["en"]}}
    status, read = request("GET", f"/api/sources/{sid}/files/{path('notes/one.md')}")
    assert read == {"version": listing["version"], "text": TEXT}

    changed = TEXT + "A third paragraph.\n"
    status, saved = request("PUT", f"/api/sources/{sid}/files/{path('notes/one.md')}",
                            {"text": changed, "patch": "more", "version": read["version"]})
    assert status == 200
    assert [p.name for p in repo.source(sid).patches] == ["0001-first-texts.patch", "0002-more.patch"]
    assert repo.read_source_file(sid, "notes/one.md", store) == (saved["version"], changed.encode())
    status, stale = request("PUT", f"/api/sources/{sid}/files/{path('notes/one.md')}",
                            {"text": "lost", "patch": "late", "version": read["version"]})
    assert status == 409 and "changed on disk" in stale["problems"][0]
    assert len(repo.source(sid).patches) == 2

    status, git = request("GET", "/api/status")
    assert git["git"] and {entry["path"] for entry in git["files"]} == {
        f"data/sources/{sid}/source.json", f"data/sources/{sid}/0001-first-texts.patch",
        f"data/sources/{sid}/0002-more.patch"}
    assert repo.check() == [] and repo.check_fetched(sid, store)[0] == 1


def conversation(*messages):
    return {"messages": list(messages)}


def text(role, words, **metadata):
    message = {"role": role, "content": [{"type": "text", "text": words}]}
    return {**message, "metadata": metadata} if metadata else message


def draft_of(content, edited):
    from wpt_dataset.conversations import content_sha256
    return {"model": "tiny", "weights": {"model.safetensors": "a" * 64}, "sha256": content_sha256(content),
            "edited": edited}


def test_the_server_sets_whether_a_model_draft_was_changed(web):
    # metadata.draft.edited is computed at save from the content, whatever
    # the editor sent
    request, repo, store = web
    sid = new_source(request, {"sft/talks.jsonl": {"format": "chat", "text": ""}})  # conversations only
    chat = f"/api/sources/{sid}/conversations/{path('sft/talks.jsonl')}"
    status, empty = request("GET", chat)
    assert status == 200 and empty["conversations"] == []

    drafted = [{"type": "text", "text": "Hello, how can I help?"}]
    confirmed = conversation(text("user", "Hello"), text("assistant", "Hello, how can I help?",
                                                         draft=draft_of(drafted, edited=True)))  # a false claim
    status, saved = request("PUT", f"{chat}/new", {"conversation": confirmed, "patch": "talks",
                                                   "version": empty["version"]})
    assert status == 200, saved
    assert repo.source(sid).data["formats"] == [{"format": "chat", "paths": ["sft/talks.jsonl"]}]

    changed = conversation(text("user", "Hello"), text("assistant", "Hi! What do you need?",
                                                       draft=draft_of(drafted, edited=False)))  # hides the edit
    status, again = request("PUT", f"{chat}/new", {"conversation": changed, "patch": "edit",
                                                   "version": saved["version"]})
    assert status == 200, again
    status, both = request("GET", chat)
    assert [c["messages"][1]["metadata"]["draft"]["edited"] for c in both["conversations"]] == [False, True]

    # a whole number the model wrote as 2.0 comes back from a browser as 2: still unchanged
    called = [{"type": "tool_call", "id": "c1", "name": "add", "arguments": {"a": 2.0}}]
    tool = conversation(text("user", "Add"), {"role": "assistant", "content": [{**called[0], "arguments": {"a": 2}}],
                                              "metadata": {"draft": draft_of(called, edited=True)}})
    status, third = request("PUT", f"{chat}/new", {"conversation": tool, "patch": "tool", "version": again["version"]})
    assert status == 200, third
    status, read = request("GET", chat)
    assert read["conversations"][2]["messages"][1]["metadata"]["draft"]["edited"] is False

    user_draft = conversation({**text("user", "Hello"), "metadata": {"draft": draft_of(drafted, edited=True)}},
                              text("assistant", "Hello"))
    status, refused = request("PUT", f"{chat}/new", {"conversation": user_draft, "patch": "x",
                                                     "version": third["version"]})
    assert status == 422  # only an assistant message is drafted by a model
    assert len(repo.source(sid).patches) == 3 and repo.check_fetched(sid, store) == (1, 0, {}, [])


def test_a_model_folder_drafts_the_next_assistant_message(web, tmp_path):
    # the draft names the folder and its weights and can be saved unedited
    request, repo, store = web
    from wpt_model import WPT, Drafter, DrafterConfig, ModelConfig, save_drafter, save_model, tokenizer_identity, \
        weight_hashes
    from wpt_tokenizer import Tokenizer

    tokenizer = Tokenizer.train([b"the quick brown fox jumps over the lazy dog"] * 4, 516)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=64, n_layer=2, global_attention=["CSA"],
                         d_model=8, kda_heads=1, kda_head_dim=8, attention_heads=1, attention_head_dim=128,
                         query_rank=4, output_groups=1, output_rank=8, index_heads=1, index_head_dim=128,
                         index_top_k=2, n_experts=2, top_k=1, expert_width=4, latent_width=4, shared_width=8,
                         engram_layers=[1], engram_order=2, engram_heads=1, engram_dim=4, engram_table_size=31)
    folder = tmp_path / "fox-model"
    save_model(WPT(config), folder, tokenizer.to_bytes(), "fox version 1")
    asked = conversation(text("user", "the quick brown fox"))
    status, refused = request("POST", "/api/drafts", {"model": str(folder), "conversation": asked})
    assert status == 422 and refused["problems"][0].startswith("DrafterMismatch"), refused  # a model needs its drafter
    # an untrained drafter, named as trained against these weights: a model folder always holds one
    save_drafter(Drafter(config, DrafterConfig.for_model(config)), folder, *tokenizer_identity(folder),
                 provenance={"target": {"weights": weight_hashes(folder)}})
    for _ in range(20):  # a random model may end its turn at once; that draft is refused, so ask again
        status, drafted = request("POST", "/api/drafts", {"model": str(folder), "conversation": asked,
                                                          "max_new_tokens": 8})
        if status == 200:
            break
    assert status == 200, drafted
    message = drafted["message"]
    assert message["role"] == "assistant" and message["content"]
    assert {k: v for k, v in message["metadata"]["draft"].items() if k != "sha256"} == {
        "model": "fox-model", "weights": weight_hashes(folder), "edited": False}

    sid = new_source(request, {"notes.txt": {"format": "text", "text": TEXT}})
    chat = f"/api/sources/{sid}/conversations/{path('sft/drafted.jsonl')}"
    status, empty = request("GET", chat)
    status, saved = request("PUT", f"{chat}/new", {"conversation": {**asked, "messages": [*asked["messages"], message]},
                                                   "patch": "drafted", "version": empty["version"]})
    assert status == 200, saved
    assert repo.source(sid).data["formats"][-1] == {"format": "chat", "paths": ["sft/drafted.jsonl"]}
    status, refused = request("POST", "/api/drafts", {"model": str(tmp_path), "conversation": asked})
    assert status == 404 and "not a WPT model folder" in refused["problems"][0]

    # the loaded model goes when its weights change, and weights that do not load (a broken checkpoint, its
    # header and drafter in order) are answered with 422, never served from the model loaded before
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from wpt_model import weight_files

    weights = folder / weight_files(folder)[0]
    with safe_open(weights, framework="pt") as opened:
        header = opened.metadata()
    save_file({"not.a.layer": torch.zeros(1)}, weights, metadata=header)
    save_drafter(Drafter(config, DrafterConfig.for_model(config)), folder, *tokenizer_identity(folder),
                 provenance={"target": {"weights": weight_hashes(folder)}})
    status, refused = request("POST", "/api/drafts", {"model": str(folder), "conversation": asked,
                                                      "max_new_tokens": 8})
    assert status == 422 and refused["problems"][0].startswith("RuntimeError"), refused


def test_a_machine_fault_while_loading_a_model_is_not_reported_as_a_broken_folder(tmp_path, monkeypatch):
    # out of memory and driver errors are RuntimeErrors too: only a fault of the folder's files is Unloadable
    import torch
    import wpt_model
    from wpt_tokenizer import Tokenizer

    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    tokenizer = Tokenizer.train([b"the quick brown fox jumps over the lazy dog"] * 4, 516).to_bytes()
    monkeypatch.setattr(wpt_model, "load_tokenizer", lambda _: tokenizer)
    asked = conversation(text("user", "hello"))

    def fails_with(error):
        def load(*_):
            raise error
        monkeypatch.setattr(wpt_model, "load_model", load)
        with pytest.raises(DraftFailed) as failed:
            Drafter().draft(str(folder), asked)
        return failed.value

    for machine in (torch.cuda.OutOfMemoryError("CUDA out of memory"), MemoryError(),
                    RuntimeError("CUDA error: an illegal memory access was encountered"),
                    RuntimeError("HIP error: out of memory")):
        refused = fails_with(machine)
        assert type(refused) is MachineFault and "not the model folder" in str(refused), (machine, refused)
    for broken in (FileNotFoundError("model.safetensors"), ValueError("config.json: not JSON"),
                   KeyError("n_layer"), RuntimeError("Error(s) in loading state_dict for WPT: size mismatch")):
        refused = fails_with(broken)
        assert type(refused) is Unloadable, (broken, refused)
    # any other exception is neither the machine's nor the folder's: unknown, with its type and text
    refused = fails_with(ZeroDivisionError("division by zero"))
    assert type(refused) is UnknownFault and "ZeroDivisionError: division by zero" in str(refused), refused
    assert "the machine failed" not in str(refused) and "not the model folder" not in str(refused)


def test_tasks_recipes_dockerfiles_and_signers_go_through_the_checks_rules(web):
    request, repo, store = web
    status, made = request("PUT", "/api/documents/dockerfile/tiny",
                           {"text": "ARG BASE=debian:trixie-slim\r\nFROM ${BASE}\r\n", "version": ""})
    assert status == 200 and repo.read("dockerfile", "tiny").text == "ARG BASE=debian:trixie-slim\nFROM ${BASE}\n"

    sid = new_source(request, {"notes.txt": {"format": "text", "text": TEXT}})
    task = {"instructions": "Print hello.", "image": "tiny",
            "limits": {"max_turns": 2, "max_tokens": 100, "command_seconds": 5}, "needs": []}
    status, tasks = request("GET", f"/api/sources/{sid}/tasks")
    status, saved = request("PUT", f"/api/sources/{sid}/tasks/{path('tasks/hello')}", {
        "task": task, "check": "#!/bin/sh\necho 1\n", "files": {"README": "say hello\n"}, "patch": "hello",
        "version": tasks["version"]})
    assert status == 200, saved
    status, tasks = request("GET", f"/api/sources/{sid}/tasks")
    assert tasks["tasks"] == {"tasks/hello": {"task": task, "check": "#!/bin/sh\necho 1\n",
                                              "files": {"README": "say hello\n"}, "kept": {}}}
    status, refused = request("PUT", f"/api/sources/{sid}/tasks/{path('tasks/other')}", {
        "task": {**task, "image": "missing"}, "check": "", "patch": "other", "version": tasks["version"]})
    assert status == 422 and "no image 'missing'" in refused["problems"][0]

    mixture = {"pools": [{"name": "en", "select": "en", "weight": 1}]}
    assert request("PUT", "/api/documents/mixture/m", {"document": mixture, "version": ""})[0] == 200
    status, refused = request("PUT", "/api/documents/mixture/half", {
        "document": {**mixture, "pools": [{**mixture["pools"][0], "weight": 0.5}]}, "version": ""})
    assert status == 422 and "pool weights add up to 0.5, not 1" in refused["problems"][0]
    assert not repo.exists("mixture", "half")
    status, schema = request("GET", "/api/schemas/model")
    assert status == 200 and "steps" in schema["properties"]

    status, refused = request("PUT", "/api/documents/lock/t", {"document": {}, "version": ""})
    assert status == 422 and "written only by `wpt-dataset build-tokenizer`" in refused["problems"][0]
    status, refused = request("PUT", "/api/documents/allowed-signers/allowed-signers",
                              {"text": "someone@example.org not-a-key\n", "version": ""})
    assert status == 422 and "line 1: no key" in refused["problems"][0]
    signer = "someone@example.org namespaces=\"git\" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample\n"
    assert request("PUT", "/api/documents/allowed-signers/allowed-signers", {"text": signer, "version": ""})[0] == 200
    status, kinds = request("GET", "/api/kinds")
    assert {k["name"]: k["names"] for k in kinds}["allowed-signers"] == ["allowed-signers"]
    status, outside = request("GET", f"/api/documents/mixture/{path('../models/m')}")
    assert status == 422 and repo.check() == []


def test_the_rating_page_lists_a_runs_attempts_and_records_a_persons_verdict(web):
    # the attempts come from a wpt-eval run's record folder in the store,
    # with the weights it measured; the verdict becomes a line of a rating
    # file of an empty source (docs/evaluation.md, "People's ratings"); an
    # attempt Docker dropped is listed as its line says, and a last line a
    # stopped run cut short is left out instead of failing the page
    request, repo, store = web
    folder, weights = "20261023T090000Z-wpt-eval-aaaaaaaa-0123abcd", {"model.safetensors": "a" * 64}
    tried = {"task": "mine-tasks-0123abcd/tasks/hello", "kind": "tasks", "index": 1, "plain": False, "dropped": False,
             "score": 1.0, "changes": "--- /dev/null\n+++ b/out.txt\n@@ -0,0 +1 @@\n+hello\n",
             "conversation": {"messages": [text("user", "Say hello."), text("assistant", "Hello.")],
                              "metadata": {"tools": {}, "looking": [], "answered_by": [], "answer": None}}}
    dropped = {"task": tried["task"], "kind": "tasks", "index": 2, "plain": False, "dropped": True,
               "error": "docker create: no space left on device",
               "conversation": {"messages": [text("user", "Say hello.")], "metadata": {}}}
    store.write_record(folder, "ids.json", json.dumps({"inputs": {"weights": weights}, "outputs": {}}).encode())
    store.append_record(folder, "attempts.jsonl", (json.dumps(tried) + "\n" + json.dumps(dropped) + "\n").encode())
    store.append_record(folder, "attempts.jsonl", json.dumps(tried).encode()[:40])
    store.flush_records()  # added lines reach the store together
    store.write_record("20261023T080000Z-wpt-train-tiny-0123abcd", "ids.json", b"{}")
    assert request("GET", "/api/runs") == (200, [folder])
    assert request("GET", f"/api/runs/{folder}/attempts") == (200, {"weights": weights, "attempts": [tried, dropped]})
    assert request("GET", "/api/runs/20261023T080000Z-wpt-train-tiny-0123abcd/attempts")[0] == 404

    sid = new_source(request, {"notes.txt": {"format": "text", "text": TEXT}})
    status, listing = request("GET", f"/api/sources/{sid}/ratings/{path('ratings/terse.jsonl')}")
    assert status == 200 and listing["ratings"] == []
    rating = {"task": tried["task"], "weights": weights, "record": folder, "attempt": 1, "plain": False,
              "verdict": "pass", "by": "A. Person", "date": "2026-10-23"}
    status, saved = request("PUT", f"/api/sources/{sid}/ratings/{path('ratings/terse.jsonl')}/new",
                            {"rating": rating, "patch": "rated", "version": listing["version"]})
    assert status == 200, saved
    assert request("GET", f"/api/sources/{sid}/ratings/{path('ratings/terse.jsonl')}")[1] == {
        "version": saved["version"], "ratings": [rating]}
    status, refused = request("PUT", f"/api/sources/{sid}/ratings/{path('ratings/terse.jsonl')}/1",
                              {"rating": {**rating, "verdict": "maybe"}, "patch": "wrong", "version": saved["version"]})
    assert status == 422 and "at verdict" in refused["problems"][0]
    assert repo.check() == []


def test_every_route_is_in_the_openapi_description_and_bodies_follow_it(web):
    request, repo, _ = web
    status, description = request("GET", "/api/openapi.json")
    assert status == 200 and description["openapi"] == "3.1.0"
    operations = {(method.upper(), pattern) for pattern, methods in description["paths"].items() for method in methods}
    assert {(r.method, r.pattern) for r in pages.routes(repo)} | {("GET", "/api/openapi.json")} == operations
    status, refused = request("PUT", "/api/documents/mixture/m", {"document": {}})
    assert status == 400 and "'version' is a required property" in refused["problems"][0]


def test_a_source_folder_without_source_json_is_listed_with_its_problem(web):
    # a stray or half-made folder never breaks the list the pages and source pickers read
    request, repo, store = web
    sid = new_source(request, {"notes.txt": {"format": "text", "text": TEXT}})
    (repo.root / "data" / "sources" / "stray-0000000a").mkdir()
    status, listed = request("GET", "/api/sources")
    assert status == 200 and [s["id"] for s in listed] == [sid, "stray-0000000a"]
    assert listed[0]["problems"] == [] and "source.json" in listed[1]["problems"][0]


def test_the_source_list_checks_each_source_once(web, monkeypatch):
    request, repo, store = web
    new_source(request, {"notes.txt": {"format": "text", "text": TEXT}})
    new_source(request, {"poem.txt": {"format": "text", "text": TEXT}}, name="poem")
    checked = []
    check_source = Repo.check_source
    monkeypatch.setattr(Repo, "check_source", lambda self, sid: checked.append(sid) or check_source(self, sid))
    status, listed = request("GET", "/api/sources")
    assert status == 200 and len(listed) == 2 and sorted(checked) == sorted(s["id"] for s in listed)
