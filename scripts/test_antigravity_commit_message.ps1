[CmdletBinding()]
param(
    [ValidateSet("small", "medium", "full", "all")]
    [string]$Phase = "all",

    [switch]$RestoreOriginalIndex = $true,
    [switch]$SkipPrompt
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & git @Arguments 2>&1 | ForEach-Object { $_.ToString() }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "git $($Arguments -join ' ') failed:`n$output"
    }
    return $output
}

function Get-RepoRoot {
    return (Invoke-Git -Arguments @("rev-parse", "--show-toplevel")).Trim()
}

function Get-AntigravityLogFile {
    $logRoot = Join-Path $env:APPDATA "Antigravity\logs"
    if (-not (Test-Path $logRoot)) {
        throw "Antigravity log root was not found at $logRoot"
    }

    $latestSession = Get-ChildItem $logRoot -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latestSession) {
        throw "No Antigravity log sessions were found in $logRoot"
    }

    $logFile = Join-Path $latestSession.FullName "window1\exthost\google.antigravity\Antigravity.log"
    if (-not (Test-Path $logFile)) {
        throw "Antigravity log file was not found at $logFile"
    }

    return $logFile
}

function Save-IndexSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$GitDir
    )

    $indexPath = Join-Path $GitDir "index"
    if (-not (Test-Path $indexPath)) {
        throw "Git index was not found at $indexPath"
    }

    $snapshotPath = Join-Path $env:TEMP ("antigravity-index-" + [guid]::NewGuid() + ".bak")
    Copy-Item -LiteralPath $indexPath -Destination $snapshotPath -Force
    return $snapshotPath
}

function Restore-IndexSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$GitDir,

        [Parameter(Mandatory = $true)]
        [string]$SnapshotPath
    )

    $indexPath = Join-Path $GitDir "index"
    Copy-Item -LiteralPath $SnapshotPath -Destination $indexPath -Force
    Invoke-Git -Arguments @("status", "--short") | Out-Null
}

function Clear-Staging {
    Invoke-Git -Arguments @("reset", "-q", "HEAD", "--", ".") | Out-Null
}

function Add-PathspecsFromHead {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Filter
    )

    $trackedPaths = Invoke-Git -Arguments @("ls-tree", "-r", "--name-only", "HEAD", "--", "scrap_prediction_v1/backend")
    $paths = $trackedPaths | Where-Object $Filter

    if (-not $paths) {
        return
    }

    $pathspecFile = Join-Path $env:TEMP ("antigravity-pathspec-" + [guid]::NewGuid() + ".txt")
    try {
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllLines($pathspecFile, $paths, $utf8NoBom)
        Invoke-Git -Arguments @("add", "-A", "--pathspec-from-file=$pathspecFile") | Out-Null
    }
    finally {
        Remove-Item -LiteralPath $pathspecFile -ErrorAction SilentlyContinue
    }
}

function Stage-Small {
    Clear-Staging
    Invoke-Git -Arguments @("add", "--", "requirements.txt") | Out-Null
}

function Stage-Medium {
    Clear-Staging
    Invoke-Git -Arguments @("add", "--", "requirements.txt") | Out-Null
    Add-PathspecsFromHead -Filter {
        $_ -match '^scrap_prediction_v1/backend/.+\.py$' -or
        $_ -eq 'scrap_prediction_v1/backend/requirements.txt'
    }
}

function Stage-Full {
    Clear-Staging
    Invoke-Git -Arguments @("add", "-A") | Out-Null
    Invoke-Git -Arguments @("reset", "-q", "HEAD", "--", "scripts/open_antigravity_repo.ps1", "scripts/test_antigravity_commit_message.ps1") | Out-Null
}

function Get-StagedSummary {
    $shortStat = Invoke-Git -Arguments @("diff", "--cached", "--shortstat")
    if ($shortStat) {
        return ($shortStat -join [Environment]::NewLine).Trim()
    }

    return "No staged changes."
}

function Get-StagedNames {
    return Invoke-Git -Arguments @("diff", "--cached", "--name-status")
}

function Show-PhaseHeader {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    Write-Host ""
    Write-Host "=== $Name ==="
    Write-Host (Get-StagedSummary)
    Write-Host ""
    Write-Host "Staged files:"
    Get-StagedNames | ForEach-Object { Write-Host "  $_" }
}

function Get-LogCursor {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LogFile
    )

    $lines = Get-Content -LiteralPath $LogFile
    return [pscustomobject]@{
        LogFile = $LogFile
        LineCount = $lines.Count
    }
}

function Get-NewCommitMessageLogLines {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Cursor
    )

    $logFile = Get-AntigravityLogFile
    $lines = Get-Content -LiteralPath $logFile

    if ($logFile -eq $Cursor.LogFile) {
        if ($lines.Count -gt $Cursor.LineCount) {
            $lines = $lines[$Cursor.LineCount..($lines.Count - 1)]
        }
        else {
            $lines = @()
        }
    }

    return $lines | Where-Object {
        $_ -match "GenerateCommitMessage" -or
        $_ -match "Failed to generate commit message"
    }
}

function Invoke-InteractiveCheck {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    Show-PhaseHeader -Name $Name
    $cursor = Get-LogCursor -LogFile (Get-AntigravityLogFile)

    if (-not $SkipPrompt) {
        Write-Host ""
        Write-Host "In Antigravity, run: Antigravity: Generate Commit Message"
        Write-Host "Then confirm whether a message is inserted in Source Control."
        [void](Read-Host "Press Enter after the attempt")
    }

    $newLines = Get-NewCommitMessageLogLines -Cursor $cursor
    $failed = $newLines | Where-Object { $_ -match "Failed to generate commit message|GenerateCommitMessage \(unknown\).*CANCELLED" }

    return [pscustomobject]@{
        Phase = $Name
        Summary = Get-StagedSummary
        Failed = [bool]$failed
        LogFile = (Get-AntigravityLogFile)
        LogLines = $newLines
    }
}

$repoRoot = Get-RepoRoot
Set-Location $repoRoot
$gitDir = (Invoke-Git -Arguments @("rev-parse", "--git-dir")).Trim()
if (-not [System.IO.Path]::IsPathRooted($gitDir)) {
    $gitDir = Join-Path $repoRoot $gitDir
}

$indexSnapshot = $null
if ($RestoreOriginalIndex) {
    $indexSnapshot = Save-IndexSnapshot -GitDir $gitDir
}

$results = New-Object System.Collections.Generic.List[object]

try {
    switch ($Phase) {
        "small" {
            Stage-Small
            [void]$results.Add((Invoke-InteractiveCheck -Name "small"))
        }
        "medium" {
            Stage-Medium
            [void]$results.Add((Invoke-InteractiveCheck -Name "medium"))
        }
        "full" {
            Stage-Full
            [void]$results.Add((Invoke-InteractiveCheck -Name "full"))
        }
        "all" {
            Stage-Small
            [void]$results.Add((Invoke-InteractiveCheck -Name "small"))

            Stage-Medium
            [void]$results.Add((Invoke-InteractiveCheck -Name "medium"))

            Stage-Full
            [void]$results.Add((Invoke-InteractiveCheck -Name "full"))
        }
    }
}
finally {
    if ($RestoreOriginalIndex -and $indexSnapshot -and (Test-Path $indexSnapshot)) {
        Restore-IndexSnapshot -GitDir $gitDir -SnapshotPath $indexSnapshot
        Remove-Item -LiteralPath $indexSnapshot -ErrorAction SilentlyContinue
    }
}

$reportPath = Join-Path $env:TEMP ("antigravity-commit-message-report-" + [guid]::NewGuid() + ".json")
$results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host ""
Write-Host "Report saved to:"
Write-Host "  $reportPath"
Write-Host ""

foreach ($result in $results) {
    $status = if ($result.Failed) { "FAILED" } else { "NO NEW FAILURE LOGGED" }
    Write-Host "$($result.Phase): $status"
    if ($result.LogLines.Count -gt 0) {
        Write-Host "  New Antigravity log lines:"
        foreach ($line in $result.LogLines) {
            Write-Host "    $line"
        }
    }
}
