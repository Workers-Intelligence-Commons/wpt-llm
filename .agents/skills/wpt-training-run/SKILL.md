---
name: wpt-training-run
description: How to start a WPT training run so it survives a closed terminal or SSH session and keeps its log. Use it before running wpt-train (or any long wpt-* command such as a large build or evaluation) on Windows, Linux or macOS, locally or over SSH, and when a run must be followed, stopped or resumed.
---

# Starting a training run

Read the `wpt` skill first for the commands and their options. This skill
is how a run is started and looked after.

## Before the run

- The dataset is built from the model recipe: `uv run wpt-dataset build
  NAME` prints its id. A first build of a tokenizer recipe writes its lock
  (`data/recipes/tokenizers/NAME.lock.json`): commit it.
- PyTorch is installed for this machine's GPU: a plain `uv sync` removes
  it, `uv run scripts/setup.py` puts it back.
- The model folder (`--out`, under `runs/`, which git ignores) is new, or
  holds a checkpoint to resume.

## Always start it detached

A program started from a terminal or an SSH session ends when the session
closes, and its output is lost. Start every run, and every command that
takes more than a few minutes, through the detach script of the machine,
from the repository folder:

| System | Command |
| --- | --- |
| Windows (PowerShell) | `.\scripts\detach.ps1 uv run wpt-train RECIPE DATASET_ID --out runs/NAME [options]` |
| Linux, macOS | `scripts/detach.sh uv run wpt-train RECIPE DATASET_ID --out runs/NAME [options]` |

On a Windows machine, use the Windows row from PowerShell, never a WSL
shell (the `wpt` skill, "Setting up a machine").

Both run the command outside the session, write everything it prints to
`runs/logs/<date-time>.log` (UTF-8, line by line), and print the process
id with how to follow and stop it:

| | Windows | Linux, macOS |
| --- | --- | --- |
| follow | `Get-Content -Wait 'runs\logs\<file>.log'` | `tail -f 'runs/logs/<file>.log'` |
| stop | `taskkill /T /F /PID <id>` | `kill <id>` |

`kill <id>` lets the run finish its step and write a checkpoint before it
exits (code 130). `taskkill /F` ends it at once: the steps since the last
checkpoint (every `--checkpoint-every` steps) are lost.

Over SSH, start the run with the script, end the SSH command, and read the
log in later commands; never keep a run attached to the connection.

## While it runs

Read the log to report progress: each evaluation line gives the step, the
training and validation loss, and the time. The run ends with
`saved runs/NAME` followed by a sample of generated text. A process that is
gone without that line has failed: the end of the log says why.

## After a crash or a stop

A crash restarts every process on its own, up to `--max-restarts` times
(default 3), from the last checkpoint; the log says `starting every process
again`. A refusal (exit code 2: missing dataset, other settings than the
checkpoint's) is never restarted.

After a stop, or once the restarts are used up, the same command with
`--resume` continues from `runs/NAME/checkpoint/`, started the same
detached way. Without `--resume`, a leftover checkpoint stops the run, so
an unfinished run is never overwritten by mistake.

## Long windows

A recipe's stages grow the window (docs/training.md, "Stages"); a stage
whose window does not fit the GPU, even with `--recompute --micro-batch
1`, needs one of these options, which change nothing in the training
(docs/training.md, "Long windows"):

- one GPU: `--segment N`, with N a multiple of 128 that fits (the memory
  of an N-token window, plus about 31 KiB per token of the window for
  watery-tardigrade), up to twice the time;
- several GPUs: `--strategy split` with `--processes N` or torchrun, each
  window's length a multiple of 128 times the number of processes;
- several GPUs whose slices still do not fit: `--strategy split` with
  `--segment N`, each process training its slice in segments of N.

Any of them may be added or changed when resuming.

## A run over several machines or sessions

With `--run NAME`, the checkpoint is kept in the store (`WPT_STORE`)
instead of the model folder: any machine using the same store continues
the run with the same command plus `--run NAME --resume`. Use it when a run
may move to another machine, or with torchrun on several machines
(docs/training.md, "Restarting after a crash").
