import hashlib
import json
import re

import pytest

from wpt_dataset import builder, manifest, tasks
from wpt_dataset.build import write_document_list
from wpt_dataset.repo import empty_source
from wpt_dataset.sources import source_tree
from wpt_dataset.tasks import EVAL
from wpt_telemetry import Telemetry

# a book of plain paragraphs, each long enough to be asked for, their starts all different
BOOK = "\n\n".join(f"Chapter {n} tells how the keeper of light number {n} climbed the stairs at dusk, lit the lamp "
                   f"with the long brass taper, and wrote the hour, the wind and the state of the sea in the "
                   f"log the company keeps for every lighthouse on the coast, before he slept."
                   for n in range(1, 40)) + "\n"
COMMIT = "Count the lamps\n\nThe keeper counts them at dusk, every night.\n"


def commit(kit, folder, message, files):
    for path, content in files.items():
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        (folder / path).write_bytes(content)
    kit.run("git", "add", "-A", cwd=folder)
    kit.run("git", "commit", "-q", "-m", message, cwd=folder)
    return kit.run("git", "rev-parse", "HEAD", cwd=folder).decode().strip()


def images(repo):
    (repo.root / "data" / "images" / "agent").mkdir(parents=True)
    (repo.root / "data" / "images" / "agent" / "Dockerfile").write_text("FROM debian:trixie-slim\n")


def task_source(repo, store):
    """An empty source made for the built tasks, its license text written and admitted, tagged finish-tasks."""
    document = empty_source("built", license="CC0-1.0", attribution="A. Person, built tasks",
                            text="CC0 1.0 Universal\n", by="A. Person", formats={"README": "text"}, tags=["finish-tasks"])
    sid = repo.create_source("mine", "built", document, {}, "first", store)
    data = repo.source(sid).data
    repo.save("source", sid, {**data, "decision": {"decision": "admitted", "by": "A. Person", "date": "2026-10-22"}},
              repo.version("source", sid))
    return sid


def books(kit, repo, store, tmp_path):
    images(repo)
    folder = tmp_path / "books"
    files = {f"book-{n}.txt": BOOK.replace("keeper", f"keeper {n}").encode() for n in range(3)}
    address, revision = kit.git_project(folder, files=files, executable=set(), symlinks={})
    sid = kit.add_source(repo, store, "example", "books", {"type": "git", "address": address, "revision": revision,
                                                           "tags": [{"tags": ["en"], "paths": ["**"]}]})
    source_tree(repo.source(sid), store)  # fetched into the store, as `check --fetch` leaves it
    return sid


def built(repo, store, kind, count, seed=1, measure=False):
    """The tasks built and the tasks read back, those a training step reads
    or, with measure, those the measures read."""
    made = builder.build(repo, store, kind, count, seed, "en", log=lambda _: None)
    sid = task_source(repo, store)
    builder.write(repo, store, sid, kind, made, repo.source_version(sid))
    # the model's dataset, its document list kept, and nothing held out
    write_document_list(store, "d" * 64, [], set(), Telemetry.off())
    found, _ = tasks.read(repo, store, "finish-tasks", 0, measure=measure, dataset="d" * 64)
    # the final reply is the answer, as text; no instructions ask for JSON
    assert not any("JSON" in task.instructions or "confidence" in task.instructions for task in found)
    return made, sorted(found, key=lambda task: task.id)


def paragraph(task):
    """The paragraph the task asks for: its quoted start and the rest the check expects."""
    quote = task.files["quote.txt"][1]
    rest = (task.check_files.get("rest.txt") or task.check_files["missing.txt"])[1]
    return quote, rest


def texts(task):
    return {path: content for path, (_, content) in {**task.files, **task.tool_files}.items()
            if path.startswith(("docs/", "web/"))}


def test_find_the_passage_tasks_ask_for_the_rest_of_a_paragraph_that_one_text_holds(kit, repo, store, tmp_path):
    # the model is given the start of a paragraph and must find the rest in
    # one of the task's texts: as files in /work, or as pages only search and
    # fetch reach; the same seed builds the same tasks again
    books(kit, repo, store, tmp_path)
    made, found = built(repo, store, "find-the-passage", 4)
    assert len(found) == 4
    for task in found:
        quote, rest = paragraph(task)
        whole = quote + b" " + rest
        assert [path for path, text in texts(task).items() if whole in text] and \
               sum(text.count(quote) for text in texts(task).values()) == 1
        pages = bool(task.tool_files)
        assert pages == ({tool["name"] for tool in task.tools} == {"search", "fetch"})
        assert len(texts(task)) == (builder.PAGES if pages else builder.GIVEN)
        assert all(len(text) <= builder.WINDOW for text in texts(task).values())
        assert not any(path.startswith("web/") for path in task.files)
        # it only answers, so no path of /work may change, and no rule gates it
        assert task.area == () and task.gates == () and b"sources_first=wpt_check.sources_first(conversation)" in task.check
        assert task.answered_in == ()  # no question to the user, so nothing to look in before one
    assert sorted(bool(task.tool_files) for task in found) == [False, False, True, True]
    assert builder.build(repo, store, "find-the-passage", 4, 1, "en", log=lambda _: None) == made


def test_evidence_removed_tasks_are_half_without_the_paragraph(kit, repo, store, tmp_path):
    # half the tasks have the passage
    # removed, and the right answer is then "not enough"; the other half can
    # be answered, so saying "not enough" is never a safe way out
    books(kit, repo, store, tmp_path)
    _, found = built(repo, store, "evidence-removed", 6)
    removed = [task for task in found if "missing.txt" in task.check_files]
    assert len(removed) == 3
    for task in found:
        quote, rest = paragraph(task)
        held = [text for text in texts(task).values() if quote in text]
        if task in removed:  # "not enough" must name the paragraph by the words of its quoted start
            words = len(re.findall(r"\w+", quote.decode()))
            assert held == [] and rest.startswith(quote) and \
                   f"not_enough(conversation, wpt_check.beside('missing.txt'), {words})".encode() in task.check
        else:
            assert len(held) == 1 and quote + b" " + rest in held[0]
        assert "not enough" in task.instructions


def test_look_or_ask_tasks_hide_the_file_name_in_the_notes_or_in_the_user_s_reply(kit, repo, store, tmp_path):
    # one missing detail, where the result goes: half the tasks write it in
    # NOTES.md, half give it only through the ask tool's reply
    books(kit, repo, store, tmp_path)
    _, found = built(repo, store, "look-or-ask", 4)
    asked = 0
    for task in found:
        name = re.search(rb"result-[0-9a-f]{6}\.txt", task.check)[0].decode()
        [(pattern, reply)] = task.replies
        assert name in reply and re.search(pattern, "Which file should I write it to?")
        notes = task.files["NOTES.md"][1].decode()
        # the user is asked after the looking tools that can read the notes (the test image has no pool)
        assert task.answered_by == ("read", "bash") and b"asked_last(conversation" in task.check
        # the answer lives in the notes, read before asking in both halves
        assert task.answered_in == ("NOTES.md",)
        # the result's file is all it may change, and the rules the check wants
        # gate it: a broken one scores the attempt 0
        assert task.area == (name,) and b"read_before_write=wpt_check.read_before_write(conversation" in task.check
        if (tasks.Variant(("ask",)),) in task.needs:
            asked += 1
            assert name not in notes and b"read_before_ask" not in task.check and task.gates == ("asked_last",)
            assert b"asked_last(conversation, '')" in task.check
        else:  # the notes, which name the file, must have been read before any question
            assert name in notes and f"read_before_ask(conversation, 'Put results in {name}.')".encode() in task.check
            # a question after the notes showed the name no longer counts
            assert f"asked_last(conversation, 'Put results in {name}.')".encode() in task.check
            assert task.gates == ("asked_last", "read_before_ask")
    assert asked == 2


def test_read_the_manual_tasks_ask_for_a_result_of_ondrik_and_half_remove_its_manual(kit, repo, store, tmp_path):
    # a result only the image's program
    # gives, with and without its manual installed; seals and knots are
    # crossed with the two halves, the instructions never name the option
    # that gives the result, and the check runs the program for it; the
    # tasks are for the measures only, their folder tagged eval by the
    # builder through the service layer, so no training step reads them
    images(repo)
    _, found = built(repo, store, "read-the-manual", 8, measure=True)
    [sid] = {task.id.split("/", 1)[0] for task in found}
    assert {"tags": [EVAL], "paths": ["read-the-manual/**"]} in repo.source(sid).data["tags"]
    assert tasks.read(repo, store, "finish-tasks", 0, dataset="d" * 64)[0] == []
    halves = {}
    for task in found:
        assert task.needs == ((tasks.Variant(("bash",)),),) and task.files == {}
        command = "press" if "the seal of the text" in task.instructions else "tie"
        assert f"[{builder.PROGRAM!r}, {command!r}, ".encode() in task.check
        assert "--turns" not in task.instructions and "--give" not in task.instructions
        halves[task.absent, command] = halves.get((task.absent, command), 0) + 1
    assert halves == {((), "press"): 2, ((), "tie"): 2, ((builder.MANUAL,), "press"): 2, ((builder.MANUAL,), "tie"): 2}


def lamps(kit, repo, store, folder, revision, since, cuts=()):
    """The id of an admitted git source of folder's history from since to revision, in the store."""
    sid = kit.add_source(repo, store, "example", "lamps", {
        "type": "git", "address": str(folder), "revision": revision, "history": {"since": since},
        "licenses": kit.WITH_HISTORY, "formats": [{"format": "code", "language": "patch", "paths": [".git/history/**"]},
                                                  {"format": "text", "paths": ["LICENSE", "*.py", "*.txt", "docs/*"]}],
        **({"cuts": list(cuts)} if cuts else {})})
    source_tree(repo.source(sid), store)
    return sid


def unpacked(task, work):
    """Write the task's /work files into the folder work; returns it."""
    for path, (_, content) in task.files.items():
        (work / path).parent.mkdir(parents=True, exist_ok=True)
        (work / path).write_bytes(content)
    return work


def test_a_commit_and_push_task_makes_work_the_repository_before_the_commit(kit, repo, store, tmp_path):
    # /work is the repository itself, checked out at the commit's parent
    # with the commit's changes in its working tree; its remote is outside
    # /work, so `git add -A` there never takes it; what the check expects is
    # the commit's tree on top of that parent. The instructions are the decided words
    # alone, and the model writes its own message, so none is given or
    # expected
    images(repo)
    folder = tmp_path / "upstream"
    folder.mkdir()
    kit.run("git", "init", "-q", cwd=folder)
    since = commit(kit, folder, "start", {"LICENSE": kit.LICENSE_TEXT.encode(), "lamps.py": b"LAMPS = 1\n"})
    head = commit(kit, folder, COMMIT, {"lamps.py": b"LAMPS = 2\n", "docs/count.md": b"# Count\n"})
    sid = lamps(kit, repo, store, folder, head, since)
    _, found = built(repo, store, "commit-and-push", 3)
    task = next(task for task in found if task.documents[0][1] == f".git/history/{head}")

    expected = json.loads(task.check_files[tasks.REPOSITORY][1])
    assert expected["tree"] == kit.run("git", "rev-parse", f"{head}^{{tree}}", cwd=folder).decode().strip()
    assert expected["remote"] == tasks.REMOTE == "/tmp/origin.git"
    assert task.instructions == ("Based on the current diffs, could you commit and push the changes in this "
                                 "repository?")
    # /work holds the repository only, its files at the top, no message to copy; the remote is outside it
    assert "message" not in expected and sorted(path for path in task.files if not path.startswith(".git/")) == \
        ["LICENSE", "docs/count.md", "lamps.py"]
    assert list(task.outside) == [tasks.REMOTE] and task.files["lamps.py"][1] == b"LAMPS = 2\n"
    work = unpacked(task, tmp_path / "work")
    kit.run("git", "add", "-A", cwd=work)
    assert kit.run("git", "diff", "--cached", "--name-status", cwd=work).decode().splitlines() == \
        ["A\tdocs/count.md", "M\tlamps.py"]
    assert kit.run("git", "write-tree", cwd=work).decode().strip() == expected["tree"]
    assert kit.run("git", "rev-parse", "HEAD^{tree}", cwd=work) == kit.run("git", "rev-parse", f"{since}^{{tree}}",
                                                                           cwd=folder)
    assert kit.run("git", "rev-parse", "HEAD", cwd=work).decode().strip() == expected["parent"]
    assert kit.run("git", "config", "remote.origin.url", cwd=work).decode().strip() == tasks.REMOTE
    remote = tmp_path / "remote"
    for path, (_, content) in task.outside[tasks.REMOTE].items():
        (remote / path).parent.mkdir(parents=True, exist_ok=True)
        (remote / path).write_bytes(content)
    assert kit.run("git", "--git-dir", str(remote), "rev-parse", "main").decode().strip() == expected["parent"]
    # one way to push per attempt: git through bash, or the push tool with the remote hidden
    assert task.needs == ((tasks.Variant(("bash",)), tasks.Variant(("bash", "push"), (tasks.REMOTE,))),)
    assert [tool["name"] for tool in task.tools] == ["push"]
    assert task.pointers == () and task.documents[0][0] == sid
    assert task.area == (".git/**",) and task.gates == ()  # git's own records: a project file changed is over-reach


def test_an_older_commit_s_task_starts_from_the_real_state_just_before_it(kit, repo, store, tmp_path):
    # every commit of the stored
    # history is a candidate; an older commit's task undoes, in the stored
    # order, every later commit newest first, then that commit, and expects
    # the commit's own tree. The commits
    # behind the newest merge are dropped before drawing, never logged
    images(repo)
    folder = tmp_path / "upstream"
    folder.mkdir()
    kit.run("git", "init", "-q", "-b", "main", cwd=folder)
    since = commit(kit, folder, "start", {"LICENSE": kit.LICENSE_TEXT.encode(), "lamps.py": b"LAMPS = 1\n"})
    kit.run("git", "checkout", "-q", "-b", "side", cwd=folder)
    commit(kit, folder, "Note the tides", {"tides.txt": b"high at noon\n"})
    kit.run("git", "checkout", "-q", "main", cwd=folder)
    hours = commit(kit, folder, "Write down the keeper's hours\n", {"docs/hours.md": b"# Hours\n"})
    kit.run("git", "merge", "-q", "--no-ff", "-m", "Merge the tides", "side", cwd=folder)
    older = commit(kit, folder, "Count two lamps\n", {"lamps.py": b"LAMPS = 2\n"})
    head = commit(kit, folder, COMMIT, {"lamps.py": b"LAMPS = 3\n", "docs/count.md": b"# Count\n"})
    lamps(kit, repo, store, folder, head, since)

    def drawn(count, log=lambda _: None):
        made = builder.build(repo, store, "commit-and-push", count, 1, log=log)
        return sorted(task["pointers"][0]["document"] for task, _, _ in made.values())

    # since, side and hours sit behind the merge: dropped before drawing, and nothing is logged about them
    logged = []
    assert drawn(5, log=logged.append) == sorted([f".git/history/{head}", f".git/history/{older}"])
    assert not [line for line in logged if hours in line or since in line or "left out" in line]
    assert len(drawn(1)) == 1 and drawn(1) == drawn(1)
    _, found = built(repo, store, "commit-and-push", 3)
    task = next(task for task in found if task.documents[0][1] == f".git/history/{older}")
    work = unpacked(task, tmp_path / "work")
    # the later commit is undone: its file is not there, and lamps.py is as the older commit left it
    assert not (work / "docs" / "count.md").exists() and (work / "lamps.py").read_bytes() == b"LAMPS = 2\n"
    assert kit.run("git", "status", "--porcelain", cwd=work).decode().splitlines() == [" M lamps.py"]
    assert kit.run("git", "rev-parse", "HEAD^{tree}", cwd=work) == kit.run("git", "rev-parse", f"{older}^^{{tree}}",
                                                                           cwd=folder)
    assert json.loads(task.check_files[tasks.REPOSITORY][1])["tree"] == \
        kit.run("git", "rev-parse", f"{older}^{{tree}}", cwd=folder).decode().strip()


def test_a_commit_whose_undo_fails_is_logged_and_the_next_one_is_drawn(kit, repo, store, tmp_path):
    # a failed undo is logged and
    # skipped, and the next candidate is drawn. A cut changes tides.txt in
    # the final tree, so the older commit that wrote it no longer undoes;
    # the later commit, which left it alone, still does
    images(repo)
    folder = tmp_path / "upstream"
    folder.mkdir()
    kit.run("git", "init", "-q", cwd=folder)
    since = commit(kit, folder, "start", {"LICENSE": kit.LICENSE_TEXT.encode(), "lamps.py": b"LAMPS = 1\n"})
    older = commit(kit, folder, "Note the tides\n", {"tides.txt": b"high at noon\n"})
    head = commit(kit, folder, COMMIT, {"lamps.py": b"LAMPS = 2\n"})
    lamps(kit, repo, store, folder, head, since, [{"path": "tides.txt", "start": 7, "end": 12,
                                                   "sha256_after": hashlib.sha256(b"high at\n").hexdigest()}])
    for seed in range(64):
        logged = []
        made = builder.build(repo, store, "commit-and-push", 1, seed, log=logged.append)
        assert [task["pointers"][0]["document"] for task, _, _ in made.values()] == [f".git/history/{head}"]
        if any(f".git/history/{older}: left out" in line and "undoing" in line for line in logged):
            break
    else:
        pytest.fail(f"no seed drew {older} first")


def test_a_stored_history_without_its_order_gives_no_task(kit, repo, store, tmp_path, monkeypatch):
    # the order is part of the stored
    # history; a history stored before it kept one gives no commit-and-push
    # task, and the builder counts such sources in one line
    images(repo)
    folder = tmp_path / "upstream"
    folder.mkdir()
    kit.run("git", "init", "-q", cwd=folder)
    since = commit(kit, folder, "start", {"LICENSE": kit.LICENSE_TEXT.encode(), "lamps.py": b"LAMPS = 1\n"})
    head = commit(kit, folder, COMMIT, {"lamps.py": b"LAMPS = 2\n"})
    read_git_history = manifest.read_git_history

    def unordered(repository, since, revision, extractor):  # the history as stored before it kept its order
        add = extractor.add
        extractor.add = lambda path, *rest: None if path == manifest.ORDER else add(path, *rest)
        return read_git_history(repository, since, revision, extractor)

    monkeypatch.setattr(manifest, "read_git_history", unordered)
    sid = lamps(kit, repo, store, folder, head, since)
    tree, _ = source_tree(repo.source(sid), store)
    assert f".git/history/{head}" in tree and manifest.ORDER not in tree
    logged = []
    assert builder.build(repo, store, "commit-and-push", 3, 1, log=logged.append) == {}
    assert [line for line in logged if manifest.ORDER in line] == [
        f"1 git sources with a history have no {manifest.ORDER} in the store and gave no task (their history was "
        f"stored before histories kept their order; they give tasks once fetched under a new checksum)"]


def test_a_commit_whose_source_changed_is_refused_until_it_is_built_again(kit, repo, store, tmp_path):
    # a task names the source checksum it was built from: a source fixed at
    # another revision gives the model another history than its check expects
    images(repo)
    folder = tmp_path / "upstream"
    folder.mkdir()
    kit.run("git", "init", "-q", cwd=folder)
    since = commit(kit, folder, "start", {"LICENSE": kit.LICENSE_TEXT.encode(), "lamps.py": b"LAMPS = 1\n"})
    head = commit(kit, folder, COMMIT, {"lamps.py": b"LAMPS = 2\n"})
    sid = lamps(kit, repo, store, folder, head, since)
    task = task_source(repo, store)
    document = {"schema": 3, "instructions": "Commit and push.", "image": "agent", "limits": builder.LIMITS,
                "needs": [["bash"]], "pointers": [{"for": "repository", "source": sid, "document": f".git/history/{head}",
                                                   "checksum": "sha256:" + "0" * 64}]}
    with pytest.raises(tasks.InvalidFile, match=rf"the repository pointer: {sid} now has the checksum "
                                                rf"{repo.source(sid).data['checksum']}, not sha256:0+, which the task "
                                                rf"was built from; build the task again"):
        builder.write(repo, store, task, "commit-and-push",
                      {"commit-and-push/a": (document, builder.check_text("right = 1"), {})},
                      repo.source_version(task))
    assert repo.source(task).patches == []  # nothing written


def test_the_builder_reports_the_act_and_check_shares_of_a_task_source(kit, repo, store, tmp_path):
    # each kind acts, checks or does
    # both (commit-and-push acts, read-the-manual does both, the others
    # check); a task source's shares count its built tasks by kind and sum
    # to 1
    books(kit, repo, store, tmp_path)  # the agent image too, which read-the-manual tasks run in
    folder = tmp_path / "upstream"
    folder.mkdir()
    kit.run("git", "init", "-q", cwd=folder)
    since = commit(kit, folder, "start", {"LICENSE": kit.LICENSE_TEXT.encode(), "lamps.py": b"LAMPS = 1\n"})
    head = commit(kit, folder, COMMIT, {"lamps.py": b"LAMPS = 2\n"})
    lamps = kit.add_source(repo, store, "example", "lamps", {
        "type": "git", "address": str(folder), "revision": head, "history": {"since": since},
        "licenses": kit.WITH_HISTORY, "formats": [{"format": "code", "language": "patch", "paths": [".git/history/**"]},
                                                  {"format": "text", "paths": ["LICENSE", "*.py"]}],
        "tags": [{"tags": ["code"], "paths": ["**"]}]})
    source_tree(repo.source(lamps), store)
    sid = task_source(repo, store)
    assert builder.shares(repo, store, sid) == {"tasks": dict.fromkeys(builder.KINDS, 0), "act": 0.0, "check": 0.0,
                                                "both": 0.0}
    for kind, count in (("find-the-passage", 3), ("commit-and-push", 1), ("read-the-manual", 2)):
        builder.write(repo, store, sid, kind, builder.build(repo, store, kind, count, 1, "en", log=lambda _: None),
                      repo.source_version(sid))
    assert builder.shares(repo, store, sid) == {
        "tasks": {"find-the-passage": 3, "evidence-removed": 0, "look-or-ask": 0, "commit-and-push": 1,
                  "read-the-manual": 2},
        "act": pytest.approx(1 / 6), "check": 0.5, "both": pytest.approx(1 / 3)}
