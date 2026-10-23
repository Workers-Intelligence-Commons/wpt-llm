import dataclasses
import hashlib
import json
import random
import re
import subprocess
import time

import pytest

from wpt_agent import (AGENT, Limits, Sandbox, SandboxError, arranged, attempt, attempts, changes, docker, needed,
                       offer, opening, plain, score)
from wpt_agent.loop import ANSWERED, CONFIDENCE, CONTEXT, FULL, TOKENS, TURNS
from wpt_agent.sandbox import ROOT
from wpt_agent.tools import ASK, NOT_KNOWN, OTHER_NAMES, TOOLS, call
from wpt_dataset.manifest import EXECUTABLE, FILE, SYMLINK
from wpt_dataset.tasks import ASK as ASK_NAME, BUILT_IN, REMOTE, Task, Variant
from wpt_tokenizer import Tokenizer, chat

TOKENIZER = Tokenizer([])  # bytes only: any text encodes
LIMITS = Limits(memory=256, disk=16, processes=64, cpus=1.0)


def sandbox(image, files=None, limits=LIMITS):
    return Sandbox(image, task().files if files is None else files, user=AGENT, limits=limits)


def task(**limits):
    """The example task; limits sets max_turns, max_tokens, command_seconds,
    or any other field of the Task."""
    made = Task("example/hello", "Say hello in out.txt.", "tiny", limits.pop("max_turns", 8),
                limits.pop("max_tokens", 10_000), limits.pop("command_seconds", 10), {"notes.txt": (FILE, b"start\n")},
                b"#!/bin/sh\ngrep -q hello out.txt && echo 1 || echo 0\n")
    return dataclasses.replace(made, **limits)


def turn(*blocks) -> list[int]:
    """The tokens a model writes for an assistant message: its content and
    the closing <|im_end|>."""
    ids = chat.encode(TOKENIZER, {"messages": [{"role": "assistant", "content": list(blocks)}]})[0]
    start = ids.index(chat.IM_START) + len(TOKENIZER.encode(b"assistant\n")) + 1
    return ids[start:ids.index(chat.IM_END) + 1]


def bash(command, n=1):
    return {"type": "tool_call", "id": f"call-{n}", "name": "bash", "arguments": {"command": command}}


class Scripted:
    """A model whose row r writes writes[r](limit) at each call, whatever it
    read; called as attempts' start, it gives itself as the writer."""

    def __init__(self, *writes):
        self.writes, self.read = writes, []

    def __call__(self, ids, rows, stops):
        self.read.append((ids, rows))
        return self

    def write(self, inputs, limits):
        return {row: self.writes[row](limit) for row, limit in limits.items()}


SURE = {"type": "text", "text": "50"}  # a reply to the loop's confidence question


def scripted(*turns):
    """A model that writes the given turns, one per call, then SURE when
    the loop asks for its confidence after its final reply."""
    pending = [*turns, turn(SURE)]
    return Scripted(lambda count: pending.pop(0)[:count])


def results(done):
    return [block["content"][0]["text"] for message in done.conversation["messages"] if message["role"] == "tool"
            for block in message["content"]]


def test_the_tools_offered_are_the_names_the_task_format_allows():
    # wpt_dataset refuses a declared tool that takes one of these names
    assert [tool["name"] for tool in TOOLS] == list(BUILT_IN) and ASK["name"] == ASK_NAME


def test_tool_calls_run_in_the_container_and_the_check_scores_what_they_left(image):
    # the model's calls change the container's files, each result comes back
    # as the next turn, and only the tokens the model wrote are learned
    turns = [turn(bash("cat notes.txt && echo hi > out.txt")),
             turn({"type": "tool_call", "id": "call-1", "name": "edit",
                   "arguments": {"path": "out.txt", "old_string": "hi", "new_string": "hello"}}),
             turn({"type": "tool_call", "id": "call-1", "name": "write",
                   "arguments": {"path": "deep/er/new.txt", "content": "x"}},
                  {"type": "tool_call", "id": "call-2", "name": "read", "arguments": {"path": "out.txt"}}),
             turn({"type": "text", "text": "Done."})]
    with sandbox(image) as box:
        done = attempt(task(), box, TOKENIZER, scripted(*turns), result_bytes=1000)
        assert score(box, task().check, 10, done.conversation) == (1.0, "")

    assert done.ended == ANSWERED
    assert results(done) == ["start\n", "edited out.txt", "wrote 1 bytes to deep/er/new.txt", "hello\n"]
    assert [i for i, learned in zip(done.ids, done.learn) if learned] == [i for t in [*turns, turn(SURE)] for i in t]
    assert done.ids[:len(chat.prompt(TOKENIZER, opening(task())))] == chat.prompt(TOKENIZER, opening(task()))
    shown = chat.show(TOKENIZER, done.ids)
    assert shown.count("<|im_start|>assistant\n") == 5 and "<tool_response>\nhello\n\n</tool_response>" in shown


def test_limits_stop_a_command_cut_a_long_result_and_end_the_attempt(image):
    # a command past its seconds is stopped and the model is told; a long
    # result keeps its start and its end; the last allowed turn's calls run
    turns = [turn(bash("sleep 60", 1), bash("seq 1 1000", 2))]
    started = time.monotonic()
    with sandbox(image, {}) as box:
        done = attempt(task(max_turns=1, command_seconds=1), box, TOKENIZER, scripted(*turns), result_bytes=30)
    assert time.monotonic() - started < 30
    assert done.ended == TURNS
    stopped, cut = results(done)
    assert stopped == "[stopped after 1 seconds]"
    printed = "".join(f"{n}\n" for n in range(1, 1001))
    assert cut == f"{printed[:15]}\n[cut: {len(printed) - 30} bytes]\n{printed[-15:]}"


def test_a_command_counts_as_stopped_only_when_it_ran_out_of_time(image):
    # exiting 124 is the command's own code, while one that ignores TERM is
    # killed after the grace and still reported as stopped
    with sandbox(image, {}) as box:
        assert call(box, "bash", {"command": "exit 124"}, 10, 1000, plain(task())) == ("[exit code 124]", True)
        started = time.monotonic()
        assert call(box, "bash", {"command": "trap '' TERM; sleep 60"}, 1, 1000, plain(task())) == ("[stopped after 1 seconds]", True)
    assert time.monotonic() - started < 30


def test_an_argument_with_a_nul_or_a_lone_surrogate_fails_for_the_model():
    # a NUL or a lone surrogate (valid JSON, not UTF-8) gets a failed result
    # before anything runs, never an exception that would stop the step
    for text in ("a\u0000b", "a\ud800b"):
        for name, arguments in [("bash", {"command": text}), ("write", {"path": "out.txt", "content": text}),
                                ("write", {"path": text, "content": "x"}),
                                ("edit", {"path": "out.txt", "old_string": text, "new_string": "x"}),
                                ("edit", {"path": "out.txt", "old_string": "x", "new_string": text})]:
            result, failed = call(None, name, arguments, 10, 1000, plain(task()))
            assert failed and result.startswith(f"{name}: "), result


def test_a_command_that_prints_without_end_never_fills_the_host_s_memory(image):
    # the output is cut while it is read: `yes` writes gigabytes in two
    # seconds, and the result still holds only its start and its end
    with sandbox(image, {}) as box:
        text, failed = call(box, "bash", {"command": "timeout 2 yes; echo done"}, 10, 1000, plain(task()))
    cut = int(re.search(r"\[cut: (\d+) bytes\]", text)[1])
    printed = cut + 1000  # bytes: "y\n" many times, then "done\n"
    assert not failed and cut > 10_000_000
    assert text == "y\n" * 250 + f"\n[cut: {cut} bytes]\n" + ("y\n" * ((printed - 5) // 2) + "done\n")[-500:]


def test_edit_refuses_a_file_that_is_not_utf8_and_leaves_it_as_it_was(image):
    with sandbox(image, {"data.bin": (FILE, b"caf\xe9 au lait\n")}) as box:
        text, failed = call(box, "edit", {"path": "data.bin", "old_string": "lait", "new_string": "miel"}, 10, 1000, plain(task()))
        assert failed and text == "data.bin is not UTF-8 text (byte 3), so edit cannot rewrite it"
        assert box.run(["cat", "data.bin"], seconds=10, limit=100).stdout == b"caf\xe9 au lait\n"


def test_the_model_s_commands_run_without_root_inside_the_limits(image):
    # the container cannot grow past its sizes: memory, processes, CPU time,
    # and the space of the only folders it can write; it keeps only the
    # three capabilities root unpacks the starting files with (CHOWN,
    # DAC_OVERRIDE, FOWNER: 0xb), the model's commands hold none of them,
    # and nothing in it can gain privileges (no setuid); the starting files
    # belong to the model's user
    script = ("id -u; cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/pids.max /sys/fs/cgroup/cpu.max; "
              "grep -E '^(CapEff|CapBnd|NoNewPrivs):' /proc/self/status | cut -f2; "
              "stat -c '%u:%g %a' /work sub sub/a.txt; "
              "touch /etc/x 2>/dev/null || echo read-only; "
              "head -c 20M /dev/zero > big 2>/dev/null || echo full; du -m big | cut -f1")
    with sandbox(image, {"sub/a.txt": (FILE, b"a\n")}) as box:
        text, failed = call(box, "bash", {"command": script}, 10, 1000, plain(task()))
    assert text == (f"1000\n{256 << 20}\n64\n100000 100000\n0000000000000000\n000000000000000b\n1\n"
                    "1000:1000 755\n1000:1000 755\n1000:1000 644\nread-only\nfull\n16\n")


def test_finished_background_processes_never_fill_the_process_limit(image):
    # the container's init collects every process that ended, so a model that
    # left more finished background processes than the limit can still run
    with sandbox(image, {}, Limits(memory=256, disk=16, processes=16, cpus=1.0)) as box:
        for _ in range(3):
            call(box, "bash", {"command": "for i in $(seq 10); do (true &); done"}, 10, 1000, plain(task()))
        assert call(box, "bash", {"command": "echo still running"}, 10, 1000, plain(task())) == ("still running\n", False)


def test_a_model_that_fills_its_process_limit_ends_its_attempt_and_is_scored(image):
    # processes left running in their own sessions, more than the limit
    # allows: no command can start any more, so the attempt ends there,
    # docker's own error never comes back to the model as a tool result,
    # and the check still scores what /work holds
    limits = Limits(memory=256, disk=16, processes=64, cpus=1.0)
    turns = [turn(bash("echo hello > out.txt; for i in $(seq 100); do setsid sleep 1000 & done")),
             turn(bash("echo still here"))]
    with sandbox(image, {}, limits) as box:
        done = attempt(task(command_seconds=5), box, TOKENIZER, scripted(*turns), result_bytes=1000)
        assert score(box, task().check, 10, done.conversation) == (1.0, "")
    assert done.ended == FULL
    assert len(results(done)) == 1 and "OCI" not in results(done)[0]
    assert done.conversation["messages"][-1]["role"] == "assistant"


def test_a_model_whose_files_fill_the_memory_ends_its_attempt_and_is_scored(image):
    # files in /work and /tmp, together more than the memory holds, until no
    # command can start: the attempt ends, and is scored like any other.
    # Running out of memory may also stop the container, whose /work is
    # then lost, so the score depends on which process the kernel ended
    limits = Limits(memory=64, disk=48, processes=64, cpus=1.0)
    filling = turn(bash("echo hello > out.txt; head -c 40M /dev/zero > big; head -c 8M /dev/zero > /tmp/$RANDOM"))
    with sandbox(image, {}, limits) as box:
        done = attempt(task(command_seconds=5, max_turns=20), box, TOKENIZER, Scripted(lambda count: filling[:count]),
                       result_bytes=1000)
        scored, why = score(box, task().check, 10, done.conversation)
    assert done.ended == FULL and scored in (0.0, 1.0) and why == ""
    assert not any("OCI" in text for text in results(done))


def test_docker_s_own_failures_never_pass_for_the_command_s(image):
    # a container that stopped, which is not the model's doing, raises for
    # every command and for scoring, instead of returning docker's text as
    # a command's output; docker's exit code names a failure it gave no text for
    with sandbox(image, {}) as box:
        with pytest.raises(docker.DockerFailed, match="^docker exec: exit 137$"):
            docker.run("exec", box.container, "sh", "-c", "exit 137")
        subprocess.run(["docker", "kill", box.container], capture_output=True, check=True)
        with pytest.raises(SandboxError, match=f"^container {box.container}: docker exec: .*not running"):
            call(box, "bash", {"command": "echo hi"}, 10, 1000, plain(task()))
        with pytest.raises(SandboxError, match=f"^container {box.container}: docker exec: .*not running"):
            box.run(["cat", "notes.txt"], seconds=10, limit=1000)
        with pytest.raises(SandboxError, match=f"^container {box.container} is not running$"):
            score(box, task().check, 10, {})


def test_a_file_put_into_tmp_leaves_tmp_writable_by_everyone(image):
    # the check is put in a new folder of /tmp; /tmp itself keeps its mode
    with Sandbox(image, {}, user=ROOT, limits=LIMITS) as box:
        box.put("/tmp/wpt-check-x/check", b"#!/bin/sh\necho 1\n", EXECUTABLE)
        ran = box.run(["sh", "-c", "stat -c %a /tmp; /tmp/wpt-check-x/check"], seconds=10, limit=100)
    assert ran.stdout == b"1777\n1\n"


def test_a_copy_whose_second_docker_cannot_start_leaves_no_process_behind(image, monkeypatch):
    # when the process that unpacks the copy cannot start, the one reading
    # /work is stopped and waited for, and the failure names docker's error
    started, real = [], docker.start

    def start(*args, **streams):
        if args[0] == "exec":
            raise docker.DockerFailed("docker exec: cannot start")
        started.append(real(*args, **streams))
        return started[-1]

    with sandbox(image) as box, sandbox(image, {}) as other:
        box.freeze()
        monkeypatch.setattr(docker, "start", start)
        with pytest.raises(SandboxError, match="^docker exec: cannot start$"):
            box.copy_work(other)
    assert [process.returncode is not None for process in started] == [True]


def test_a_failing_close_never_hides_the_error_that_left_the_block(image, monkeypatch):
    # the error the caller must handle stays the one raised inside the block;
    # the container left behind is named in a note on it
    real = docker.run

    def refusing(*args, **options):
        if args[0] == "rm":
            raise docker.DockerFailed("docker rm: refused")
        return real(*args, **options)

    boxes = [sandbox(image, {}), sandbox(image, {})]
    left = [box.container for box in boxes]
    try:
        monkeypatch.setattr(docker, "run", refusing)
        with pytest.raises(ValueError) as raised:
            with boxes[0]:
                raise ValueError("inside")
        assert raised.value.__notes__ == [f"container {left[0]} is left behind: docker rm: refused"]
        with pytest.raises(SandboxError, match=f"^container {left[1]} is left behind: docker rm: refused$"):
            with boxes[1]:
                pass  # with nothing leaving the block, the failing close is the error
    finally:
        monkeypatch.undo()
        subprocess.run(["docker", "rm", "-f", "-v", *left], capture_output=True)


def test_a_sandbox_that_cannot_start_keeps_its_error_when_removing_it_fails_too(image, monkeypatch):
    # the starting files cannot be unpacked and the container cannot be
    # removed: the unpacking error stays the one raised, naming in a note
    # the container left behind
    real = docker.run

    def refusing(*args, **options):
        if args[0] in ("exec", "rm"):
            raise docker.DockerFailed(f"docker {args[0]}: refused")
        return real(*args, **options)

    monkeypatch.setattr(docker, "run", refusing)
    with pytest.raises(SandboxError) as raised:
        sandbox(image)
    monkeypatch.undo()
    [note] = raised.value.__notes__
    left = re.fullmatch(r"container (\w+) is left behind: docker rm: refused", note)
    subprocess.run(["docker", "rm", "-f", "-v", left[1]], capture_output=True, check=True)
    assert str(raised.value) == "docker exec: refused"


def test_the_check_sees_only_the_attempt_s_work_folder_once_its_processes_are_stopped(image):
    # nothing the model starts or leaves outside /work reaches the check:
    # the attempt's processes stop when it ends, and the check runs in a
    # fresh container holding a copy of /work
    check = b"#!/bin/sh\ntest -e out.txt && ! test -e /tmp/planted && ! test -e /tmp/later && echo 1 || echo 0\n"
    with sandbox(image) as box:
        call(box, "bash", {"command": "echo hello > out.txt; echo planted > /tmp/planted; "
                                      "(sleep 1; echo later > /tmp/later) > /dev/null 2>&1 &"}, 10, 1000,
             plain(task()))
        assert score(box, check, 10, {}) == (1.0, "")
        state = subprocess.run(["docker", "inspect", "-f", "{{.State.Paused}}", box.container],
                               capture_output=True, text=True, check=True).stdout
    assert state == "true\n"


def test_an_attempt_ends_at_its_token_limit_and_at_the_models_context():
    # no call runs once a limit cuts a turn, and the attempt never holds
    # more tokens than the model can read
    endless = Scripted(lambda count: [ord("a")] * count)
    start = len(chat.prompt(TOKENIZER, opening(task())))
    done = attempt(task(max_tokens=7), None, TOKENIZER, endless, result_bytes=10)
    assert (done.ended, sum(done.learn), len(done.ids)) == (TOKENS, 7, start + 7)
    done = attempt(task(), None, TOKENIZER, endless, context=start + 5, result_bytes=10)
    assert (done.ended, len(done.ids)) == (CONTEXT, start + 5)
    assert done.conversation["messages"][-1] == {"role": "assistant", "content": [{"type": "text", "text": "aaaaa"}]}


def test_attempts_written_together_read_the_opening_once_and_end_apart():
    # one row answers in two short pieces, the other writes without end
    # until the task's max_tokens: each ends on its own, and the model read
    # the task's opening once, for both
    pieces = [[ord("o"), ord("k")], [chat.IM_END], [ord("7"), chat.IM_END]]
    model = Scripted(lambda count: pieces.pop(0)[:count], lambda count: [ord("a")] * min(count, 3))
    answered, cut = attempts(task(max_tokens=10), [None, None], TOKENIZER, model, result_bytes=10)
    prompt = chat.prompt(TOKENIZER, opening(task()))
    assert model.read == [(prompt, 2)]
    assert answered.ended == ANSWERED and answered.ids[:len(prompt) + 3] == prompt + [ord("o"), ord("k"), chat.IM_END]
    assert answered.conversation["metadata"]["answer"] == {"answer": "ok", "confidence": 0.07}
    assert cut.ended == TOKENS and cut.ids == prompt + [ord("a")] * 10 and sum(cut.learn) == 10


def test_an_attempt_opens_with_the_full_request_or_with_its_one_line_form():
    # the measure of terse against full requests gives the same task in
    # either form (maintainer's decision, 2026-10-23); the model reads the
    # form asked for as the user's message
    both = task(terse="Say hello.")
    [full, terse] = [attempt(both, None, TOKENIZER, scripted(turn({"type": "text", "text": "Done."})),
                             result_bytes=10, terse=form) for form in (False, True)]
    assert [done.conversation["messages"][0]["content"][0]["text"] for done in (full, terse)] == [
        "Say hello in out.txt.", "Say hello."]
    assert terse.ids[:len(chat.prompt(TOKENIZER, opening(both, terse=True)))] == chat.prompt(
        TOKENIZER, opening(both, terse=True))
    with pytest.raises(ValueError, match="no one-line form"):
        opening(task(), terse=True)


def test_a_check_that_prints_no_score_gives_none_and_says_why(image):
    for check, scored in [
        (b"#!/bin/sh\necho passed\n", (None, "the check printed 'passed', not a number from 0 to 1 (exit code 0)")),
        (b"#!/bin/sh\necho 1.5; echo oops >&2; exit 3\n",
         (None, "the check printed '1.5', not a number from 0 to 1 (exit code 3): oops")),
        (b"#!/bin/sh\nsleep 60\n", (None, "the check was stopped after 1 seconds")),
        (b"#!/bin/sh\necho '  0.25'\nexit 1\n", (0.25, "")),
        (b"#!/bin/sh\necho 0.5\nexit 124\n", (0.5, "")),  # timeout's code, but the check was not stopped
    ]:
        with sandbox(image, {}) as box:
            assert score(box, check, 1 if b"sleep" in check else 10, {}) == scored


def test_the_ask_tool_answers_from_the_replies_written_in_advance_and_the_attempt_goes_on():
    # the first reply whose pattern is found in the question is given, a
    # question no pattern matches gets "I don't know that", and the attempt
    # goes on after each; a task without replies offers no ask tool
    asking = task(replies=(("(?i)which colou?r", "Blue."), ("colou?r", "The other one.")))

    def ask(question, n=1):
        return {"type": "tool_call", "id": f"call-{n}", "name": "ask", "arguments": {"question": question}}

    done = attempt(asking, None, TOKENIZER, scripted(turn(ask("Which colour do you want?"), ask("Any colour?", 2)),
                                                     turn(ask("What now?")), turn({"type": "text", "text": "Blue."})),
                   result_bytes=1000)
    assert results(done) == ["Blue.", "The other one.", NOT_KNOWN] and done.ended == ANSWERED
    assert "ask" in [tool["name"] for tool in opening(asking)["tools"]]
    assert "ask" not in [tool["name"] for tool in opening(task())["tools"]]
    assert call(None, "ask", {"question": "Which colour?"}, 10, 1000, plain(task())) == (
        "unknown tool 'ask'; the tools are read, write, edit, bash", True)


def test_the_final_reply_is_the_answer_and_the_confidence_comes_from_the_question_after_it():
    # the model writes text only: its final reply is the answer, then the
    # loop asks CONFIDENCE and builds metadata.answer from the reply, a
    # number from 0 to 100 divided by 100 (docs/chat-format.md, "The answer");
    # any other reply gives no answer, which the check library scores 0
    final = turn({"type": "thinking", "text": "It is written."}, {"type": "text", "text": "Wrote hello into out.txt."})
    thought = {"type": "thinking", "text": "Fairly sure."}
    for reply, kept in [([{"type": "text", "text": "80"}], 0.8), ([{"type": "text", "text": " 72.5\n"}], 0.725),
                        ([thought, {"type": "text", "text": "100"}], 1.0), ([{"type": "text", "text": "0"}], 0.0),
                        ([{"type": "text", "text": "101"}], None), ([{"type": "text", "text": "80%"}], None),
                        ([{"type": "text", "text": "About 80."}], None), ([{"type": "text", "text": "-5"}], None),
                        ([], None), ([bash("echo 80")], None)]:
        done = attempt(task(), None, TOKENIZER, scripted(final, turn(*reply)), result_bytes=10)
        assert done.ended == ANSWERED
        assert done.conversation["metadata"]["answer"] == (
            None if kept is None else {"answer": "Wrote hello into out.txt.", "confidence": kept})
    messages = done.conversation["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[2]["content"] == [{"type": "text", "text": CONFIDENCE}]
    assert chat.show(TOKENIZER, done.ids).endswith(
        f"<|im_start|>user\n{CONFIDENCE}<|im_end|>\n<|im_start|>assistant\n" + chat.show(TOKENIZER, turn(*reply)))
    learned = [i for i, kept in zip(done.ids, done.learn) if kept]
    assert learned == final + turn(*reply)  # the question is read, never learned


def test_an_attempt_without_a_final_reply_or_a_whole_confidence_has_no_answer():
    # a limit before the final reply, or one that cuts the reply to the
    # confidence question, leaves no answer: the attempt scores 0
    unknown = {"type": "tool_call", "id": "call-1", "name": "nothing", "arguments": {}}  # runs without a sandbox
    done = attempt(task(max_turns=1), None, TOKENIZER, scripted(turn(unknown)), result_bytes=10)
    assert done.ended == TURNS and done.conversation["metadata"] == {
        "tools": {name: name for name in ["read", "write", "edit", "bash"]}, "looking": ["read", "bash"],
        "answered_by": [], "answer": None}
    final = turn({"type": "text", "text": "Done."})
    sure = turn({"type": "text", "text": "90"})
    done = attempt(task(max_tokens=len(final) + 1), None, TOKENIZER, scripted(final, sure), result_bytes=10)
    assert done.ended == TOKENS and done.conversation["metadata"]["answer"] is None
    assert done.conversation["messages"][-1] == {"role": "assistant", "content": [{"type": "text", "text": "9"}]}


def test_the_opening_lists_the_offered_tools_that_look_by_their_real_names():
    # one definition of looking (maintainer's decision, 2026-10-22): read,
    # bash and each declared tool marked looks; the check library reads the
    # list from the conversation, whatever names the attempt saw
    fetch = {"name": "fetch", "description": "Fetch a page's text.", "program": "/usr/local/bin/fetch",
             "parameters": {"type": "object"}, "names": ["get_page"], "looks": True}
    put = {"name": "put_file", "description": "Put a file.", "program": "/usr/local/bin/put",
           "parameters": {"type": "object"}}
    made = task(tools=(put,))
    for seed in range(20):
        chosen = offer(made, [fetch], random.Random(seed))
        offered = set(chosen.names.values())
        looking = opening(made, chosen)["metadata"]["looking"]
        assert sorted(looking) == sorted(offered & {"read", "bash", "fetch"})


def test_a_needed_tool_is_always_offered_and_the_others_vary_with_the_seed():
    # each attempt gets one variant of every needed tool under its own
    # name; the other tools of the task and of the image's pool are left
    # out or kept, renamed or not, in any order, and the same seed gives
    # the same offer again; the mapping names each real tool
    pool = [{"name": "fetch", "description": "Fetch a page's text.", "program": "/usr/local/bin/fetch",
             "parameters": {"type": "object"}, "names": ["get_page"]}]
    put = {"name": "put_file", "description": "Put a file.", "program": "/usr/local/bin/put",
           "parameters": {"type": "object"}}
    needing = task(needs=((Variant(("write",)), Variant(("put_file",))), (Variant(("ask",)),)), replies=(("x", "y"),),
                   tools=(put,))
    seen, renamed, orders = set(), set(), set()
    for seed in range(200):
        chosen = offer(needing, pool, random.Random(seed))
        assert chosen == offer(needing, pool, random.Random(seed))
        assert [tool["name"] for tool in chosen.tools] == list(chosen.names)
        real = list(chosen.names.values())
        assert real.count("ask") == 1 and chosen.names["ask"] == "ask"
        assert sorted(name for name in real if name in ("write", "put_file")) in (["put_file"], ["write"])
        for shown, actual in chosen.names.items():
            assert shown == actual or shown in OTHER_NAMES.get(actual, []) + (["get_page"] if actual == "fetch" else [])
            renamed |= {actual} if shown != actual else set()
        seen |= set(real)
        orders.add(tuple(chosen.names))
    assert seen == {"read", "write", "put_file", "edit", "bash", "ask", "fetch"}
    assert renamed >= {"read", "edit", "bash", "fetch"} and len(orders) > 100
    assert all(len(offer(needing, pool, random.Random(seed)).tools) < 7 for seed in range(3))
    assert list(plain(needing).names) == ["read", "write", "edit", "bash", "ask", "put_file"]


def test_each_attempt_gets_one_way_of_a_need_with_the_files_that_way_hides():
    # maintainer's decision, 2026-10-22: a commit-and-push attempt pushes one
    # way, the seeded choice of a variant: git through bash, no push tool
    # offered; or the push tool with the remote, a folder outside /work,
    # hidden among the tools' files under its own path
    push = {"name": "push", "description": "Push.", "program": "/usr/local/bin/push", "parameters": {"type": "object"}}
    remote = {"HEAD": (FILE, b"ref: refs/heads/main\n"), "config": (FILE, b"[core]\n")}
    pushing = task(needs=((Variant(("bash",)), Variant(("bash", "push"), (REMOTE,))),), tools=(push,),
                   outside={REMOTE: remote})
    ways = set()
    for seed in range(40):
        chosen = offer(pushing, [], random.Random(seed))
        found = arranged(pushing, chosen)
        offered = set(chosen.names.values())
        assert "bash" in offered and ("push" in offered) == bool(chosen.hidden) and found.files == pushing.files
        moved = {f"tmp/origin.git/{path}": entry for path, entry in remote.items()}
        assert (found.tool_files, found.outside) == ((moved, {}) if chosen.hidden else ({}, {REMOTE: remote}))
        ways.add("push" in offered)
    assert ways == {True, False}


def test_the_needed_tools_are_the_varied_attempt_s_variants_under_their_real_names():
    # an attempt offered only what it needs gets the same variant of each
    # need, and so the same hidden files, as the varied attempt drawn from a
    # generator in the same state, and nothing renamed, shuffled or added;
    # the task's absent folders come with the arrangement either way
    push = {"name": "push", "description": "Push.", "program": "/usr/local/bin/push", "parameters": {"type": "object"}}
    pushing = task(needs=((Variant(("bash",)), Variant(("bash", "push"), (REMOTE,))), (Variant(("read",)),)),
                   tools=(push,), outside={REMOTE: {"HEAD": (FILE, b"ref: refs/heads/main\n")}},
                   absent=("/usr/share/doc/thing",))
    pool = [{"name": "fetch", "description": "Fetch.", "program": "/usr/local/bin/fetch",
             "parameters": {"type": "object"}}]
    ways = set()
    for seed in range(40):
        varied, only = offer(pushing, pool, random.Random(seed)), needed(pushing, random.Random(seed))
        expected = ["bash", "push", "read"] if varied.hidden else ["bash", "read"]
        assert only.names == {name: name for name in expected} and [t["name"] for t in only.tools] == expected
        assert only.hidden == varied.hidden and set(only.names) <= set(varied.names.values())
        assert arranged(pushing, only) == arranged(pushing, varied)
        assert arranged(pushing, only).absent == ("/usr/share/doc/thing",)
        ways.add(only.hidden)
    assert len(ways) == 2


def test_an_attempt_s_changes_to_work_are_a_diff_naming_what_is_not_text():
    # what the rating page shows of an attempt: each changed text file as a
    # unified diff, a binary or large file named with its size, a change of
    # mode named, nothing for a file left as it was; the whole cut past its limit
    def left(mode, content):
        return mode, content, len(content), hashlib.sha256(content).hexdigest()

    before = {"same.txt": (FILE, b"one\n"), "edit.txt": (FILE, b"a\nb\n"), "gone.txt": (FILE, b"bye\n"),
              "run.sh": (FILE, b"echo hi\n"), "data.bin": (FILE, b"\0\1")}
    after = {"same.txt": left(FILE, b"one\n"), "edit.txt": left(FILE, b"a\nc"), "run.sh": left(EXECUTABLE, b"echo hi\n"),
             "data.bin": left(FILE, b"\0\2"), "new.txt": left(FILE, b"hello\n"), "link": left(SYMLINK, b"new.txt"),
             "big.txt": (FILE, None, 1 << 20, "0" * 64)}
    text = changes(before, after)
    assert text == ("big.txt: added, not shown (1048576 bytes, not UTF-8 text or more than 65536)\n"
                    "data.bin: changed, not shown (2 bytes, not UTF-8 text or more than 65536)\n"
                    "--- a/edit.txt\n+++ b/edit.txt\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n\\ No newline at end of file\n"
                    "--- a/gone.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n"
                    "--- /dev/null\n+++ b/link\n@@ -0,0 +1 @@\n+-> new.txt\n"
                    "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+hello\n"
                    f"mode of run.sh: {FILE} -> {EXECUTABLE}\n")
    assert changes(before, {path: left(*entry) for path, entry in before.items()}) == ""
    cut = changes(before, after, limit=40)
    assert cut.startswith(text[:40]) and cut.endswith(f" more bytes of changes]\n") and len(cut) < 100


def test_the_files_an_attempt_left_in_work_are_read_with_their_modes_and_hashes(image):
    # what changes() compares: every file and link in /work at the end, a
    # file larger than the limit hashed but not kept
    with sandbox(image, {"notes.txt": (FILE, b"start\n")}) as box:
        call(box, "bash", {"command": "echo hello > out.txt; chmod +x out.txt; head -c 300 /dev/zero > big; "
                                      "mkdir -p sub; ln -s ../out.txt sub/link; rm notes.txt"}, 10, 1000, plain(task()))
        box.freeze()
        found = box.work_files(100)
    assert found == {"out.txt": (EXECUTABLE, b"hello\n", 6, hashlib.sha256(b"hello\n").hexdigest()),
                     "big": (FILE, None, 300, hashlib.sha256(bytes(300)).hexdigest()),
                     "sub/link": (SYMLINK, b"../out.txt", 10, hashlib.sha256(b"../out.txt").hexdigest())}


def test_attempts_offered_other_tools_share_the_start_of_their_openings():
    # the model reads once what the openings share, then each row the rest
    # of its own, so every attempt holds exactly its own opening
    offers = [offer(task(), [], random.Random(seed)) for seed in (1, 2)]
    assert offers[0].names != offers[1].names
    model = Scripted(lambda count: [chat.IM_END], lambda count: [chat.IM_END])
    done = attempts(task(), [None, None], TOKENIZER, model, result_bytes=10, offers=offers)
    prompts = [chat.prompt(TOKENIZER, opening(task(), chosen)) for chosen in offers]
    [(shared, rows)] = model.read
    assert rows == 2 and all(prompt[:len(shared)] == shared for prompt in prompts) and prompts[0] != prompts[1]
    assert [one.ids[:len(prompt) + 1] for one, prompt in zip(done, prompts)] == [
        prompt + [chat.IM_END] for prompt in prompts]
    assert [one.conversation["metadata"]["tools"] for one in done] == [chosen.names for chosen in offers]


def test_a_tool_s_program_reads_the_files_hidden_from_the_model_s_own_tools(image):
    # a task's tool is answered by a program in the container, which reads
    # its arguments as JSON and the task's tool_files; the model's own
    # commands can neither list nor read those files
    program = b"#!/bin/sh\nid -u; id -g; cat /tmp/wpt-tools/corpus/a.txt; cat; echo; echo made > made.txt\n"
    search = {"name": "search", "description": "Search the notes.", "program": "/tmp/wpt-test/search",
              "parameters": {"type": "object"}}
    failing = {**search, "name": "broken", "program": "/tmp/wpt-test/broken"}
    searching = task(tools=(search, failing), tool_files={"corpus/a.txt": (FILE, b"hidden text\n")})
    with Sandbox(image, searching.files, user=AGENT, limits=LIMITS, tool_files=searching.tool_files) as box:
        box.put("/tmp/wpt-test/search", program, EXECUTABLE)  # stands for a program the image holds
        box.put("/tmp/wpt-test/broken", b"#!/bin/sh\necho no such page >&2; exit 3\n", EXECUTABLE)
        turns = [turn({"type": "tool_call", "id": "call-1", "name": "search", "arguments": {"query": "caf\u00e9"}},
                      {"type": "tool_call", "id": "call-2", "name": "broken", "arguments": {}},
                      bash("cat /tmp/wpt-tools/corpus/a.txt; ls /tmp/wpt-tools; cat made.txt", 3)),
                 turn({"type": "text", "text": "Done."})]
        done = attempt(searching, box, TOKENIZER, scripted(*turns), result_bytes=1000)
    found, broken, seen = results(done)
    assert found == '1000\n1001\nhidden text\n{"query": "caf\\u00e9"}\n'
    assert broken == "no such page"
    assert seen.count("Permission denied") == 2 and "hidden" not in seen and "made\n" in seen
    errors = [block.get("error", False) for m in done.conversation["messages"] if m["role"] == "tool"
              for block in m["content"]]
    assert errors == [False, True, False]  # bash ends with cat made.txt, which works


def test_the_check_reads_the_attempt_s_conversation_next_to_it(image):
    # the check scores the trace and the answer too, from the
    # conversation file beside it, which maps the names the model saw
    renamed = next(chosen for chosen in (offer(task(), [], random.Random(seed)) for seed in range(50))
                   if any(shown != actual for shown, actual in chosen.names.items()))
    shown, actual = next((shown, actual) for shown, actual in renamed.names.items() if shown != actual)
    check = (f'#!/bin/sh\nc="$(dirname "$0")/conversation.json"\ngrep -qF \'"{shown}": "{actual}"\' "$c" && '
             f'grep -qF \'"answer": {{"answer": "Nothing to change.", "confidence": 0.25}}\' "$c" '
             f'&& echo 0.25 || echo 0\n')
    end = turn({"type": "text", "text": "Nothing to change."})
    with sandbox(image, {}) as box:
        done = attempt(task(), box, TOKENIZER, scripted(end, turn({"type": "text", "text": "25"})), result_bytes=10,
                       offer=renamed)
        assert done.conversation["metadata"]["tools"] == renamed.names
        assert score(box, check.encode(), 10, done.conversation) == (0.25, "")
