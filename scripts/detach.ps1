<#
Run a command on this Windows machine so it keeps going after SSH disconnects.

Usage, from the repository folder:
    .\scripts\detach.ps1 uv run wpt-train <dataset id>

Programs started from an SSH session end when the session closes. This starts
the command through Windows' process-creation service (WMI) instead, so it
lives outside the session. Output goes to runs\logs\<date-time>.log.
#>
# The command arrives as plain arguments ($args), so none of its options (such
# as python's -c) can be mistaken for an option of this script.
$Command = $args
if (-not $Command) { throw 'usage: .\scripts\detach.ps1 <command> [arguments...]' }

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$logs = Join-Path $root 'runs\logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
# Create the log only if no run started in the same second has it
# (CreateNew refuses an existing file); otherwise add -1, -2...
$stamp = Join-Path $logs ('{0:yyyyMMdd-HHmmss}' -f (Get-Date))
$log = "$stamp.log"
for ($n = 1; ; $n++) {
    try { [IO.File]::Open($log, 'CreateNew').Close(); break }
    catch [IO.IOException] { $log = "$stamp-$n.log" }
}

# Start the program by its real file: Windows refuses to run a program through
# a symbolic link (such as winget's ...\WinGet\Links\uv.exe) in a process
# created this way, although `where` still finds it.
$program = Get-Command $Command[0] -CommandType Application -ErrorAction Stop | Select-Object -First 1
$item = Get-Item -LiteralPath $program.Source
$Command = @($(if ($item.LinkType -eq 'SymbolicLink') { $item.Target } else { $program.Source })) + @($Command | Select-Object -Skip 1)

# quote arguments that contain spaces or quotes, for cmd.exe
$line = ($Command | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join ' '
# /s: cmd strips only the outer quotes and keeps the rest literally.
$cmd = "cmd.exe /d /s /c `"$line > `"$log`" 2>&1`""
# A WMI-created process does not inherit this session's environment (PATH,
# user variables), so pass it along. PYTHONUNBUFFERED: Python writes each
# line to the log at once instead of in blocks. PYTHONIOENCODING: Python writes
# the log in UTF-8; otherwise it uses cp1252, which garbles accents and fails
# outright on characters such as emoji.
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$variables = [string[]](Get-ChildItem Env: | ForEach-Object { "$($_.Name)=$($_.Value)" })
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ EnvironmentVariables = $variables }
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmd; CurrentDirectory = $root; ProcessStartupInformation = $startup
}
if ($result.ReturnValue -ne 0) { throw "could not start the command (Win32_Process.Create returned $($result.ReturnValue))" }

"started: $line"
"process: $($result.ProcessId)"
"follow:  Get-Content -Wait '$log'"
"stop:    taskkill /T /F /PID $($result.ProcessId)"
