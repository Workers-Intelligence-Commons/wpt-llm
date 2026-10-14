# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Copy this working tree, committed or not, to the Windows GPU machine.

Usage:
    uv run scripts/sync.py [--host USER@HOST] [--dest WINDOWS_FOLDER]

The machine comes from --host / --dest, else from WPT_SYNC_HOST /
WPT_SYNC_DEST in the environment or in .env at the repository root
(git-ignored; copy .env.example to start one).

Sends every file git sees (tracked and untracked, minus .gitignore'd ones
such as .venv and runs/), so data files go too. Files that an earlier sync
created there but that no longer exist here are deleted; anything else in
the destination (its .venv, runs/) is never touched. The list of synced
files is kept there as .sync-manifest.
"""

import argparse
import base64
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
MANIFEST = ".sync-manifest"
ARCHIVE = ".sync.tar.gz"

# Runs on Windows: unpack, then delete what the previous manifest listed but the new one lacks.
UNPACK = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location -LiteralPath '{dest}'
$old = if (Test-Path -LiteralPath '{manifest}') {{ Get-Content -LiteralPath '{manifest}' -Encoding utf8 }} else {{ @() }}
tar -xzf '{archive}'
if ($LASTEXITCODE -ne 0) {{ throw "tar failed with exit code $LASTEXITCODE" }}
Remove-Item -LiteralPath '{archive}'
$new = Get-Content -LiteralPath '{manifest}' -Encoding utf8
$stale = $old | Where-Object {{ $_ -notin $new }}
foreach ($path in $stale) {{
    if (Test-Path -LiteralPath $path) {{ Remove-Item -LiteralPath $path; "deleted $path" }}
}}
"synced $($new.Count) files to {dest}"
"""


def read_env_file(path):
    """KEY=VALUE lines; blank lines and # comments are skipped."""
    if not path.exists():
        return {}
    pairs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            pairs[key.strip()] = value.strip().strip("'\"")
    return pairs


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout
    # --cached also lists files deleted from the disk but not yet from git
    return sorted(p for p in out.decode().split("\0") if p and (ROOT / p).is_file())


def ssh_powershell(host, script):
    """Run a PowerShell script on host; base64 avoids every quoting layer."""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    return subprocess.run(["ssh", host, f"pwsh -NoProfile -NonInteractive -EncodedCommand {encoded}"]).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", help="SSH destination, e.g. user@machine (WPT_SYNC_HOST)")
    parser.add_argument("--dest", help="folder on Windows, e.g. C:/Users/user/Projets/wpt-llm (WPT_SYNC_DEST)")
    args = parser.parse_args()
    settings = read_env_file(ENV_FILE) | os.environ
    args.host = args.host or settings.get("WPT_SYNC_HOST")
    args.dest = args.dest or settings.get("WPT_SYNC_DEST")
    missing = [name for name, value in [("WPT_SYNC_HOST", args.host), ("WPT_SYNC_DEST", args.dest)] if not value]
    if missing:
        sys.exit(f"missing {', '.join(missing)}: pass --host/--dest, or copy .env.example to .env and fill it in")

    files = tracked_files()
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / MANIFEST
        manifest.write_text("\n".join(files) + "\n", encoding="utf-8")
        archive = Path(tmp) / ARCHIVE
        with tarfile.open(archive, "w:gz") as tar:
            for path in files:
                tar.add(ROOT / path, arcname=path)
            tar.add(manifest, arcname=MANIFEST)
        print(f"packed {len(files)} files ({archive.stat().st_size / 1e6:.1f} MB)", flush=True)

        if ssh_powershell(args.host, f"New-Item -ItemType Directory -Force -Path '{args.dest}' | Out-Null"):
            sys.exit("could not create the destination folder")
        if subprocess.run(["scp", "-q", str(archive), f"{args.host}:/{args.dest}/{ARCHIVE}"]).returncode:
            sys.exit("copy failed; the destination is unchanged")
    script = UNPACK.format(dest=args.dest, manifest=MANIFEST, archive=ARCHIVE)
    if ssh_powershell(args.host, script):
        sys.exit("unpacking failed; files may be partly updated, run the sync again")


if __name__ == "__main__":
    main()
