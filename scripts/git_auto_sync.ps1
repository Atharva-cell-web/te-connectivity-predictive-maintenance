[CmdletBinding()]
param(
    [string]$Message,
    [switch]$DryRun,
    [switch]$SkipPull,
    [switch]$SkipPush,
    [switch]$UseCurrentStaging
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

function Get-GitDir {
    $gitDir = (Invoke-Git -Arguments @("rev-parse", "--git-dir")).Trim()
    if (-not [System.IO.Path]::IsPathRooted($gitDir)) {
        $gitDir = Join-Path (Get-Location) $gitDir
    }

    return $gitDir
}

function Save-IndexSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$GitDir
    )

    $snapshotPath = Join-Path $env:TEMP ("git-index-" + [guid]::NewGuid() + ".bak")
    Copy-Item -LiteralPath (Join-Path $GitDir "index") -Destination $snapshotPath -Force
    return $snapshotPath
}

function Restore-IndexSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$GitDir,

        [Parameter(Mandatory = $true)]
        [string]$SnapshotPath
    )

    Copy-Item -LiteralPath $SnapshotPath -Destination (Join-Path $GitDir "index") -Force
    Invoke-Git -Arguments @("status", "--short") | Out-Null
}

$repoRoot = (Invoke-Git -Arguments @("rev-parse", "--show-toplevel")).Trim()
Set-Location $repoRoot

$branch = (Invoke-Git -Arguments @("branch", "--show-current")).Trim()
if (-not $branch) {
    throw "Could not detect the current branch."
}

$gitDir = Get-GitDir
$indexSnapshot = $null

try {
    if ($DryRun) {
        $indexSnapshot = Save-IndexSnapshot -GitDir $gitDir
    }

    if (-not $UseCurrentStaging) {
        Invoke-Git -Arguments @("add", "-A") | Out-Null
    }

    $stagedFiles = Invoke-Git -Arguments @("diff", "--cached", "--name-only")
    $stagedFiles = $stagedFiles | Where-Object { $_.Trim() }

    if (-not $stagedFiles -or $stagedFiles.Count -eq 0) {
        Write-Host "No staged changes to commit."
        return
    }

    $summary = Invoke-Git -Arguments @("diff", "--cached", "--shortstat")
    $summary = ($summary -join " ").Trim()

    if (-not $Message) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $Message = "auto sync: $timestamp"
    }

    Write-Host "Repository : $repoRoot"
    Write-Host "Branch     : $branch"
    Write-Host "Commit     : $Message"
    if ($summary) {
        Write-Host "Changes    : $summary"
    }
    Write-Host "Files      : $($stagedFiles.Count)"

    foreach ($file in $stagedFiles | Select-Object -First 20) {
        Write-Host "  $file"
    }

    if ($stagedFiles.Count -gt 20) {
        Write-Host "  ..."
    }

    if ($DryRun) {
        Write-Host ""
        Write-Host "Dry run only. No commit, pull, or push was executed."
        return
    }

    Invoke-Git -Arguments @("commit", "-m", $Message) | Out-Null

    if (-not $SkipPull) {
        Invoke-Git -Arguments @("pull", "--rebase", "origin", $branch) | Out-Null
    }

    if (-not $SkipPush) {
        Invoke-Git -Arguments @("push", "origin", $branch) | Out-Null
    }

    $head = (Invoke-Git -Arguments @("rev-parse", "--short", "HEAD")).Trim()
    Write-Host ""
    Write-Host "Auto sync finished successfully."
    Write-Host "Commit hash: $head"
}
finally {
    if ($DryRun -and $indexSnapshot -and (Test-Path $indexSnapshot)) {
        Restore-IndexSnapshot -GitDir $gitDir -SnapshotPath $indexSnapshot
        Remove-Item -LiteralPath $indexSnapshot -ErrorAction SilentlyContinue
    }
}

