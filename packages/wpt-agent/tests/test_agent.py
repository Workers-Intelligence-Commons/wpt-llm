import re
import subprocess
import time

import pytest

from wpt_agent import AGENT, Limits, Sandbox, SandboxError, attempt, docker, opening, score
from wpt_agent.loop import ANSWERED, CONTEXT, FULL, TOKENS, TURNS
from wpt_agent.sandbox import ROOT
from wpt_agent.tools import call
from wpt_dataset.manifest import EXECUTABLE, FILE
from wpt_dataset.tasks import Task
from wpt_tokenizer import Tokenizer, chat

TOKENIZER = Tokenizer([])  # bytes only: any text encodes
LIMITS = Limits(memory=256, disk=16, processes=64, cpus=1.0)


def sandbox(image, files=None, limits=LIMITS):
    return Sandbox(image, task().files if files is None else files, user=AGENT, limits=limits)


def task(**limits):
    return Task("example/hello", "Say hello in out.txt.", "tiny", limits.get("max_turns", 8),
                limits.get("max_tokens", 10_000), limits.get("command_seconds", 10), {"notes.txt": (FILE, b"start\n")},
                b"#!/bin/sh\ngrep -q hello out.txt && echo 1 || echo 0\n")


def turn(*blocks) -> list[int]:
    """The tokens a model writes for an assistant message: its content and
    the closing <|im_end|>."""
    ids = chat.encode(TOKENIZER, {"messages": [{"role": "assistant", "content": list(blocks)}]})[0]
    start = ids.index(chat.IM_START) + len(TOKENIZER.encode(b"assistant\n")) + 1
    return ids[start:ids.index(chat.IM_END) + 1]


def bash(command, n=1):
    return {"type": "tool_call", "id": f"call-{n}", "name": "bash", "arguments": {"command": command}}


def scripted(*turns):
    """A model that writes the given turns, one per call, whatever it read."""
    pending = list(turns)
    return lambda ids, count: iter(pending.pop(0)[:count])


def results(done):
    return [block["content"][0]["text"] for message in done.conversation["messages"] if message["role"] == "tool"
            for block in message["content"]]


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
        done = attempt(task(), box, TOKENIZER, scripted(*turns), context=100_000, result_bytes=1000)
        assert score(box, task().check, 10) == (1.0, "")

    assert done.ended == ANSWERED
    assert results(done) == ["start\n", "edited out.txt", "wrote 1 bytes to deep/er/new.txt", "hello\n"]
    assert [i for i, learned in zip(done.ids, done.learn) if learned] == [i for t in turns for i in t]
    assert done.ids[:len(chat.prompt(TOKENIZER, opening(task())))] == chat.prompt(TOKENIZER, opening(task()))
    shown = chat.show(TOKENIZER, done.ids)
    assert shown.count("<|im_start|>assistant\n") == 4 and "<tool_response>\nhello\n\n</tool_response>" in shown


def test_limits_stop_a_command_cut_a_long_result_and_end_the_attempt(image):
    # a command past its seconds is stopped and the model is told; a long
    # result keeps its start and its end; the last allowed turn's calls run
    turns = [turn(bash("sleep 60", 1), bash("seq 1 1000", 2))]
    started = time.monotonic()
    with sandbox(image, {}) as box:
        done = attempt(task(max_turns=1, command_seconds=1), box, TOKENIZER, scripted(*turns), context=100_000,
                       result_bytes=30)
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
        assert call(box, "bash", {"command": "exit 124"}, 10, 1000) == ("[exit code 124]", True)
        started = time.monotonic()
        assert call(box, "bash", {"command": "trap '' TERM; sleep 60"}, 1, 1000) == ("[stopped after 1 seconds]", True)
    assert time.monotonic() - started < 30


def test_an_argument_with_a_nul_or_a_lone_surrogate_fails_for_the_model():
    # a NUL or a lone surrogate (valid JSON, not UTF-8) gets a failed result
    # before anything runs, never an exception that would stop the step
    for text in ("a\u0000b", "a\ud800b"):
        for name, arguments in [("bash", {"command": text}), ("write", {"path": "out.txt", "content": text}),
                                ("write", {"path": text, "content": "x"}),
                                ("edit", {"path": "out.txt", "old_string": text, "new_string": "x"}),
                                ("edit", {"path": "out.txt", "old_string": "x", "new_string": text})]:
            result, failed = call(None, name, arguments, 10, 1000)
            assert failed and result.startswith(f"{name}: "), result


def test_a_command_that_prints_without_end_never_fills_the_host_s_memory(image):
    # the output is cut while it is read: `yes` writes gigabytes in two
    # seconds, and the result still holds only its start and its end
    with sandbox(image, {}) as box:
        text, failed = call(box, "bash", {"command": "timeout 2 yes; echo done"}, 10, 1000)
    cut = int(re.search(r"\[cut: (\d+) bytes\]", text)[1])
    printed = cut + 1000  # bytes: "y\n" many times, then "done\n"
    assert not failed and cut > 10_000_000
    assert text == "y\n" * 250 + f"\n[cut: {cut} bytes]\n" + ("y\n" * ((printed - 5) // 2) + "done\n")[-500:]


def test_edit_refuses_a_file_that_is_not_utf8_and_leaves_it_as_it_was(image):
    with sandbox(image, {"data.bin": (FILE, b"caf\xe9 au lait\n")}) as box:
        text, failed = call(box, "edit", {"path": "data.bin", "old_string": "lait", "new_string": "miel"}, 10, 1000)
        assert failed and text == "data.bin is not UTF-8 text (byte 3), so edit cannot rewrite it; change it with bash"
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
        text, failed = call(box, "bash", {"command": script}, 10, 1000)
    assert text == (f"1000\n{256 << 20}\n64\n100000 100000\n0000000000000000\n000000000000000b\n1\n"
                    "1000:1000 755\n1000:1000 755\n1000:1000 644\nread-only\nfull\n16\n")


def test_finished_background_processes_never_fill_the_process_limit(image):
    # the container's init collects every process that ended, so a model that
    # left more finished background processes than the limit can still run
    with sandbox(image, {}, Limits(memory=256, disk=16, processes=16, cpus=1.0)) as box:
        for _ in range(3):
            call(box, "bash", {"command": "for i in $(seq 10); do (true &); done"}, 10, 1000)
        assert call(box, "bash", {"command": "echo still running"}, 10, 1000) == ("still running\n", False)


def test_a_model_that_fills_its_process_limit_ends_its_attempt_and_is_scored(image):
    # processes left running in their own sessions, more than the limit
    # allows: no command can start any more, so the attempt ends there,
    # docker's own error never comes back to the model as a tool result,
    # and the check still scores what /work holds
    limits = Limits(memory=256, disk=16, processes=64, cpus=1.0)
    turns = [turn(bash("echo hello > out.txt; for i in $(seq 100); do setsid sleep 1000 & done")),
             turn(bash("echo still here"))]
    with sandbox(image, {}, limits) as box:
        done = attempt(task(command_seconds=5), box, TOKENIZER, scripted(*turns), context=100_000, result_bytes=1000)
        assert score(box, task().check, 10) == (1.0, "")
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
        done = attempt(task(command_seconds=5, max_turns=20), box, TOKENIZER, lambda ids, count: iter(filling[:count]),
                       context=100_000, result_bytes=1000)
        scored, why = score(box, task().check, 10)
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
            call(box, "bash", {"command": "echo hi"}, 10, 1000)
        with pytest.raises(SandboxError, match=f"^container {box.container}: docker exec: .*not running"):
            box.run(["cat", "notes.txt"], seconds=10, limit=1000)
        with pytest.raises(SandboxError, match=f"^container {box.container} is not running$"):
            score(box, task().check, 10)


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
                                      "(sleep 1; echo later > /tmp/later) > /dev/null 2>&1 &"}, 10, 1000)
        assert score(box, check, 10) == (1.0, "")
        state = subprocess.run(["docker", "inspect", "-f", "{{.State.Paused}}", box.container],
                               capture_output=True, text=True, check=True).stdout
    assert state == "true\n"


def test_an_attempt_ends_at_its_token_limit_and_at_the_models_context():
    # no call runs once a limit cuts a turn, and the attempt never holds
    # more tokens than the model can read
    endless = lambda ids, count: iter([ord("a")] * count)
    start = len(chat.prompt(TOKENIZER, opening(task())))
    done = attempt(task(max_tokens=7), None, TOKENIZER, endless, context=100_000, result_bytes=10)
    assert (done.ended, sum(done.learn), len(done.ids)) == (TOKENS, 7, start + 7)
    done = attempt(task(), None, TOKENIZER, endless, context=start + 5, result_bytes=10)
    assert (done.ended, len(done.ids)) == (CONTEXT, start + 5)
    assert done.conversation["messages"][-1] == {"role": "assistant", "content": [{"type": "text", "text": "aaaaa"}]}


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
            assert score(box, check, 1 if b"sleep" in check else 10) == scored
