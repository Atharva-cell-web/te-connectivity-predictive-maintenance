[CmdletBinding()]
param(
    [string]$Message,
    [switch]$DryRun,
    [switch]$SkipPull,
    [switch]$SkipPush,
    [switch]$UseCurrentStaging,
    [string]$PullRemoteName = "origin",
    [string[]]$PushRemoteNames = @("origin", "vishh70"),
    [string]$PrimaryGitHubUsername = "atharvap2004",
    [string]$MirrorRemoteName = "vishh70",
    [string]$MirrorGitHubUsername = "Vishh70",
    [string]$GitHubRepo = "te-connectivity-predictive-maintenance"
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

function Invoke-GitAllowFailure {
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

    return [PSCustomObject]@{
        Output   = $output
        ExitCode = $exitCode
    }
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

function Clear-RepoCredentialUsername {
    $result = Invoke-GitAllowFailure -Arguments @("config", "--local", "--unset-all", "credential.https://github.com.username")
    $outputText = ($result.Output -join " ").Trim()
    if ($result.ExitCode -ne 0 -and $outputText -and $outputText -notmatch "No such section or key") {
        throw "git config --local --unset-all credential.https://github.com.username failed:`n$($result.Output)"
    }
}

function New-RemoteSpec {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$GitHubUsername,

        [Parameter(Mandatory = $true)]
        [string]$GitHubRepo
    )

    return [PSCustomObject]@{
        Name     = $Name
        Username = $GitHubUsername
        Owner    = $GitHubUsername
        Repo     = $GitHubRepo
    }
}

function Ensure-RepoGitHubRemote {
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$RemoteSpec
    )

    $expectedRemoteUrl = "https://$($RemoteSpec.Username)@github.com/$($RemoteSpec.Owner)/$($RemoteSpec.Repo).git"
    $currentRemoteResult = Invoke-GitAllowFailure -Arguments @("remote", "get-url", $RemoteSpec.Name)
    $currentRemoteUrl = ($currentRemoteResult.Output -join "").Trim()

    if ($currentRemoteResult.ExitCode -ne 0 -or -not $currentRemoteUrl) {
        Invoke-Git -Arguments @("remote", "add", $RemoteSpec.Name, $expectedRemoteUrl) | Out-Null
        $currentRemoteUrl = $expectedRemoteUrl
    }
    elseif ($currentRemoteUrl -ne $expectedRemoteUrl) {
        Invoke-Git -Arguments @("remote", "set-url", $RemoteSpec.Name, $expectedRemoteUrl) | Out-Null
        $currentRemoteUrl = $expectedRemoteUrl
    }

    return [PSCustomObject]@{
        Name     = $RemoteSpec.Name
        Username = $RemoteSpec.Username
        Url      = $currentRemoteUrl
    }
}

function Throw-GitHubAuthError {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,

        [Parameter(Mandatory = $true)]
        [string]$RemoteName,

        [Parameter(Mandatory = $true)]
        [string]$GitHubUsername,

        [Parameter(Mandatory = $true)]
        [string]$OriginalError
    )

    $helperPath = Join-Path $RepoRoot "scripts\fix_github_push_auth.ps1"
    throw "GitHub authentication failed for remote '$RemoteName' using GitHub user '$GitHubUsername'. Run:`n  powershell -NoProfile -ExecutionPolicy Bypass -File `"$helperPath`"`nThen retry the sync command and sign in to both GitHub accounts when prompted.`n`nOriginal error:`n$OriginalError"
}

$repoRoot = (Invoke-Git -Arguments @("rev-parse", "--show-toplevel")).Trim()
Set-Location $repoRoot

$remoteSpecs = @(
    New-RemoteSpec -Name $PullRemoteName -GitHubUsername $PrimaryGitHubUsername -GitHubRepo $GitHubRepo
    New-RemoteSpec -Name $MirrorRemoteName -GitHubUsername $MirrorGitHubUsername -GitHubRepo $GitHubRepo
)

$configuredRemotes = @{}
foreach ($remoteSpec in $remoteSpecs) {
    $configuredRemote = Ensure-RepoGitHubRemote -RemoteSpec $remoteSpec
    $configuredRemotes[$configuredRemote.Name] = $configuredRemote
}

$pushRemoteNames = $PushRemoteNames | Where-Object { $_ } | Select-Object -Unique
$pushRemotes = foreach ($pushRemoteName in $pushRemoteNames) {
    if (-not $configuredRemotes.ContainsKey($pushRemoteName)) {
        throw "Push remote '$pushRemoteName' is not configured."
    }

    $configuredRemotes[$pushRemoteName]
}

if (-not $configuredRemotes.ContainsKey($PullRemoteName)) {
    throw "Pull remote '$PullRemoteName' is not configured."
}

$pullRemote = $configuredRemotes[$PullRemoteName]
Clear-RepoCredentialUsername

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
    Write-Host "Pull remote: $($pullRemote.Name) -> $($pullRemote.Url)"
    Write-Host "Push remotes:"
    foreach ($pushRemote in $pushRemotes) {
        Write-Host "  $($pushRemote.Name) -> $($pushRemote.Url)"
    }
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
        try {
            Invoke-Git -Arguments @("pull", "--rebase", $pullRemote.Name, $branch) | Out-Null
        }
        catch {
            if ($_.Exception.Message -match "permission denied|Authentication failed|403|401") {
                Throw-GitHubAuthError -RepoRoot $repoRoot -RemoteName $pullRemote.Name -GitHubUsername $pullRemote.Username -OriginalError $_.Exception.Message
            }

            throw
        }
    }

    if (-not $SkipPush) {
        foreach ($pushRemote in $pushRemotes) {
            try {
                Invoke-Git -Arguments @("push", $pushRemote.Name, $branch) | Out-Null
            }
            catch {
                if ($_.Exception.Message -match "permission denied|Authentication failed|403|401") {
                    Throw-GitHubAuthError -RepoRoot $repoRoot -RemoteName $pushRemote.Name -GitHubUsername $pushRemote.Username -OriginalError $_.Exception.Message
                }

                throw
            }
        }
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
