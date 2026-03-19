[CmdletBinding()]
param(
    [switch]$CloseRunningWindows,
    [switch]$NewWindow
)

$ErrorActionPreference = "Stop"

$editorExe = "C:\Users\Datta\AppData\Local\Programs\Antigravity\Antigravity.exe"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not (Test-Path $editorExe)) {
    throw "Antigravity executable was not found at $editorExe"
}

if ($CloseRunningWindows) {
    Get-Process Antigravity -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
}

$arguments = @()
if ($NewWindow) {
    $arguments += "--new-window"
}
$arguments += $repoRoot

Start-Process -FilePath $editorExe -ArgumentList $arguments | Out-Null

Write-Host "Opened Antigravity on:"
Write-Host "  $repoRoot"

