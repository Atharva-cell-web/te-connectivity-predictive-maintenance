[CmdletBinding()]
param(
    [string]$Message,
    [switch]$DryRun,
    [switch]$SkipPull,
    [switch]$SkipPush,
    [switch]$UseCurrentStaging,
    [string]$RemoteName = "origin",
    [string]$GitHubOwner = "Atharva-cell-web",
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

function Normalize-LegacyMirrorRemote {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PreferredRemoteName,

        [Parameter(Mandatory = $true)]
        [string]$LegacyRemoteName
    )

    $remoteResult = Invoke-GitAllowFailure -Arguments @("remote")
    if ($remoteResult.ExitCode -ne 0) {
        return
    }

    $remotes = $remoteResult.Output | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ }
    if ($remotes -contains $LegacyRemoteName -and $remotes -notcontains $PreferredRemoteName) {
        Invoke-Git -Arguments @("remote", "rename", $LegacyRemoteName, $PreferredRemoteName) | Out-Null
    }
    elseif ($remotes -contains $LegacyRemoteName -and $remotes -contains $PreferredRemoteName) {
        Invoke-Git -Arguments @("remote", "remove", $LegacyRemoteName) | Out-Null
    }
}

function New-RemoteSpec {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$GitHubOwner,

        [Parameter(Mandatory = $true)]
        [string]$GitHubRepo
    )

    return [PSCustomObject]@{
        Name  = $Name
        Owner = $GitHubOwner
        Repo  = $GitHubRepo
    }
}

function Ensure-GitHubRemote {
    param(
        [Parameter(Mandatory = $true)]
        [psobject]$RemoteSpec
    )

    $expectedRemoteUrl = "https://github.com/$($RemoteSpec.Owner)/$($RemoteSpec.Repo).git"
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
        Name  = $RemoteSpec.Name
        Owner = $RemoteSpec.Owner
        Url   = $currentRemoteUrl
    }
}

function Get-BranchSyncState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteName,

        [Parameter(Mandatory = $true)]
        [string]$Branch
    )

    $remoteRef = "refs/remotes/$RemoteName/$Branch"
    $remoteRefResult = Invoke-GitAllowFailure -Arguments @("rev-parse", "--verify", $remoteRef)

    if ($remoteRefResult.ExitCode -ne 0) {
        return [PSCustomObject]@{
            HasRemoteBranch = $false
            Ahead           = 0
            Behind          = 0
        }
    }

    $counts = (Invoke-Git -Arguments @("rev-list", "--left-right", "--count", "HEAD...$remoteRef") | Select-Object -First 1).Trim()
    $parts = $counts -split "\s+"

    return [PSCustomObject]@{
        HasRemoteBranch = $true
        Ahead           = [int]$parts[0]
        Behind          = [int]$parts[1]
    }
}

function Throw-GitHubAuthError {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,

        [Parameter(Mandatory = $true)]
        [string]$RemoteName,

        [Parameter(Mandatory = $true)]
        [string]$OriginalError
    )

    $helperPath = Join-Path $RepoRoot "scripts\fix_github_push_auth.ps1"
    throw "GitHub authentication failed for remote '$RemoteName'. Run:`n  powershell -NoProfile -ExecutionPolicy Bypass -File `"$helperPath`"`nThen retry the sync command and sign in with a GitHub account that can push to both repos.`n`nOriginal error:`n$OriginalError"
}

$PullRemoteName = $RemoteName
$PushRemoteNames = @($PullRemoteName, "vishnu")
$PrimaryGitHubOwner = $GitHubOwner
$MirrorRemoteName = "vishnu"
$MirrorGitHubOwner = "Vishh70"

$repoRoot = (Invoke-Git -Arguments @("rev-parse", "--show-toplevel")).Trim()
Set-Location $repoRoot

Normalize-LegacyMirrorRemote -PreferredRemoteName $MirrorRemoteName -LegacyRemoteName "vishh70"

$remoteSpecs = @(
    New-RemoteSpec -Name $PullRemoteName -GitHubOwner $PrimaryGitHubOwner -GitHubRepo $GitHubRepo
)

if ($MirrorRemoteName -ne $PullRemoteName) {
    $remoteSpecs += New-RemoteSpec -Name $MirrorRemoteName -GitHubOwner $MirrorGitHubOwner -GitHubRepo $GitHubRepo
}

$configuredRemotes = @{}
foreach ($remoteSpec in $remoteSpecs) {
    $configuredRemote = Ensure-GitHubRemote -RemoteSpec $remoteSpec
    $configuredRemotes[$configuredRemote.Name] = $configuredRemote
}

Clear-RepoCredentialUsername

$pushRemoteNames = $PushRemoteNames | Where-Object { $_ } | Select-Object -Unique
$pushRemotes = foreach ($pushRemoteName in $pushRemoteNames) {
    if (-not $configuredRemotes.ContainsKey($pushRemoteName)) {
        throw "Push remote '$pushRemoteName' is not configured."
    }

    $configuredRemotes[$pushRemoteName]
}

$pullRemote = $configuredRemotes[$PullRemoteName]
$branch = (Invoke-Git -Arguments @("branch", "--show-current")).Trim()
if (-not $branch) {
    throw "Could not detect the current branch."
}

$gitDir = Get-GitDir
$indexSnapshot = $null
$createdCommit = $false

try {
    if ($DryRun) {
        $indexSnapshot = Save-IndexSnapshot -GitDir $gitDir
    }

    if (-not $UseCurrentStaging) {
        Invoke-Git -Arguments @("add", "-A") | Out-Null
    }

    $stagedFiles = Invoke-Git -Arguments @("diff", "--cached", "--name-only")
    $stagedFiles = $stagedFiles | Where-Object { $_.Trim() }

    $syncStates = @{}
    $allRemoteNames = @($PullRemoteName) + $pushRemoteNames | Select-Object -Unique
    foreach ($remoteName in $allRemoteNames) {
        $syncStates[$remoteName] = Get-BranchSyncState -RemoteName $remoteName -Branch $branch
    }

    $pullSyncState = $syncStates[$PullRemoteName]
    $needsPull = -not $SkipPull -and $pullSyncState.HasRemoteBranch -and $pullSyncState.Behind -gt 0
    $pushTargets = if ($SkipPush) {
        @()
    }
    else {
        foreach ($pushRemote in $pushRemotes) {
            $pushState = $syncStates[$pushRemote.Name]
            if (-not $pushState.HasRemoteBranch -or $pushState.Ahead -gt 0) {
                $pushRemote
            }
        }
    }

    Write-Host "Repository : $repoRoot"
    Write-Host "Branch     : $branch"
    Write-Host "Pull remote: $($pullRemote.Name) -> $($pullRemote.Url)"
    if ($pullSyncState.HasRemoteBranch) {
        Write-Host "Pull state : ahead $($pullSyncState.Ahead), behind $($pullSyncState.Behind)"
    }
    else {
        Write-Host "Pull state : remote branch does not exist yet"
    }
    Write-Host "Push remotes:"
    foreach ($pushRemote in $pushRemotes) {
        $pushState = $syncStates[$pushRemote.Name]
        if ($pushState.HasRemoteBranch) {
            Write-Host "  $($pushRemote.Name) -> $($pushRemote.Url) (ahead $($pushState.Ahead), behind $($pushState.Behind))"
        }
        else {
            Write-Host "  $($pushRemote.Name) -> $($pushRemote.Url) (branch missing)"
        }
    }

    if (-not $stagedFiles -or $stagedFiles.Count -eq 0) {
        Write-Host "Commit     : none (sync only)"
        Write-Host "Files      : 0"

        if (-not $needsPull -and $pushTargets.Count -eq 0) {
            Write-Host ""
            Write-Host "Nothing to sync. Local and remote branches are already aligned."
            return
        }

        if ($DryRun) {
            Write-Host ""
            Write-Host "Dry run only. No pull or push was executed."
            return
        }
    }
    else {
        $summary = Invoke-Git -Arguments @("diff", "--cached", "--shortstat")
        $summary = ($summary -join " ").Trim()

        if (-not $Message) {
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            $Message = "auto sync: $timestamp"
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
        $createdCommit = $true
        $pushTargets = $pushRemotes
    }

    if ($needsPull) {
        try {
            Invoke-Git -Arguments @("pull", "--rebase", $pullRemote.Name, $branch) | Out-Null
        }
        catch {
            if ($_.Exception.Message -match "permission denied|Authentication failed|403|401") {
                Throw-GitHubAuthError -RepoRoot $repoRoot -RemoteName $pullRemote.Name -OriginalError $_.Exception.Message
            }

            throw
        }
    }

    foreach ($pushRemote in $pushTargets) {
        try {
            Invoke-Git -Arguments @("push", $pushRemote.Name, $branch) | Out-Null
        }
        catch {
            if ($_.Exception.Message -match "permission denied|Authentication failed|403|401") {
                Throw-GitHubAuthError -RepoRoot $repoRoot -RemoteName $pushRemote.Name -OriginalError $_.Exception.Message
            }

            throw
        }
    }

    $head = (Invoke-Git -Arguments @("rev-parse", "--short", "HEAD")).Trim()
    Write-Host ""
    Write-Host "Auto sync finished successfully."
    if ($createdCommit) {
        Write-Host "Commit hash: $head"
    }
    else {
        Write-Host "Commit hash: unchanged ($head)"
    }
}
finally {
    if ($DryRun -and $indexSnapshot -and (Test-Path $indexSnapshot)) {
        Restore-IndexSnapshot -GitDir $gitDir -SnapshotPath $indexSnapshot
        Remove-Item -LiteralPath $indexSnapshot -ErrorAction SilentlyContinue
    }
}

return

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

function Get-BranchSyncState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteName,

        [Parameter(Mandatory = $true)]
        [string]$Branch
    )

    $remoteRef = "refs/remotes/$RemoteName/$Branch"
    $remoteRefResult = Invoke-GitAllowFailure -Arguments @("rev-parse", "--verify", $remoteRef)

    if ($remoteRefResult.ExitCode -ne 0) {
        return [PSCustomObject]@{
            HasRemoteBranch = $false
            Ahead           = 0
            Behind          = 0
        }
    }

    $counts = (Invoke-Git -Arguments @("rev-list", "--left-right", "--count", "HEAD...$remoteRef") | Select-Object -First 1).Trim()
    $parts = $counts -split "\s+"

    return [PSCustomObject]@{
        HasRemoteBranch = $true
        Ahead           = [int]$parts[0]
        Behind          = [int]$parts[1]
    }
}

function Ensure-SharedRemote {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteName,

        [Parameter(Mandatory = $true)]
        [string]$GitHubOwner,

        [Parameter(Mandatory = $true)]
        [string]$GitHubRepo
    )

    $expectedRemoteUrl = "https://github.com/$GitHubOwner/$GitHubRepo.git"
    $currentRemoteResult = Invoke-GitAllowFailure -Arguments @("remote", "get-url", $RemoteName)
    $currentRemoteUrl = ($currentRemoteResult.Output -join "").Trim()

    if ($currentRemoteResult.ExitCode -ne 0 -or -not $currentRemoteUrl) {
        Invoke-Git -Arguments @("remote", "add", $RemoteName, $expectedRemoteUrl) | Out-Null
        $currentRemoteUrl = $expectedRemoteUrl
    }
    elseif ($currentRemoteUrl -ne $expectedRemoteUrl) {
        Invoke-Git -Arguments @("remote", "set-url", $RemoteName, $expectedRemoteUrl) | Out-Null
        $currentRemoteUrl = $expectedRemoteUrl
    }

    return $currentRemoteUrl
}

function Throw-GitHubAuthError {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,

        [Parameter(Mandatory = $true)]
        [string]$RemoteName,

        [Parameter(Mandatory = $true)]
        [string]$OriginalError
    )

    $helperPath = Join-Path $RepoRoot "scripts\fix_github_push_auth.ps1"
    throw "GitHub authentication failed for remote '$RemoteName'. Run:`n  powershell -NoProfile -ExecutionPolicy Bypass -File `"$helperPath`"`nThen retry the sync command and sign in with a GitHub account that has collaborator access to the shared repo.`n`nOriginal error:`n$OriginalError"
}

$repoRoot = (Invoke-Git -Arguments @("rev-parse", "--show-toplevel")).Trim()
Set-Location $repoRoot
$remoteUrl = Ensure-SharedRemote -RemoteName $RemoteName -GitHubOwner $GitHubOwner -GitHubRepo $GitHubRepo
Clear-RepoCredentialUsername

$branch = (Invoke-Git -Arguments @("branch", "--show-current")).Trim()
if (-not $branch) {
    throw "Could not detect the current branch."
}

$gitDir = Get-GitDir
$indexSnapshot = $null
$createdCommit = $false

try {
    if ($DryRun) {
        $indexSnapshot = Save-IndexSnapshot -GitDir $gitDir
    }

    if (-not $UseCurrentStaging) {
        Invoke-Git -Arguments @("add", "-A") | Out-Null
    }

    $stagedFiles = Invoke-Git -Arguments @("diff", "--cached", "--name-only")
    $stagedFiles = $stagedFiles | Where-Object { $_.Trim() }
    $syncState = Get-BranchSyncState -RemoteName $RemoteName -Branch $branch
    $needsPull = -not $SkipPull -and $syncState.HasRemoteBranch -and $syncState.Behind -gt 0
    $needsPush = -not $SkipPush -and ((-not $syncState.HasRemoteBranch) -or $syncState.Ahead -gt 0)

    Write-Host "Repository : $repoRoot"
    Write-Host "Branch     : $branch"
    Write-Host "Remote     : $RemoteName -> $remoteUrl"
    if ($syncState.HasRemoteBranch) {
        Write-Host "Sync state : ahead $($syncState.Ahead), behind $($syncState.Behind)"
    }
    else {
        Write-Host "Sync state : remote branch does not exist yet"
    }

    if (-not $stagedFiles -or $stagedFiles.Count -eq 0) {
        Write-Host "Commit     : none (sync only)"
        Write-Host "Files      : 0"

        if (-not $needsPull -and -not $needsPush) {
            Write-Host ""
            Write-Host "Nothing to sync. Local and remote are already aligned."
            return
        }

        if ($DryRun) {
            Write-Host ""
            Write-Host "Dry run only. No pull or push was executed."
            return
        }
    }
    else {
        $summary = Invoke-Git -Arguments @("diff", "--cached", "--shortstat")
        $summary = ($summary -join " ").Trim()

        if (-not $Message) {
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            $Message = "auto sync: $timestamp"
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
        $createdCommit = $true
    }

    if (-not $SkipPull -and ($needsPull -or ($createdCommit -and $syncState.HasRemoteBranch))) {
        try {
            Invoke-Git -Arguments @("pull", "--rebase", $RemoteName, $branch) | Out-Null
        }
        catch {
            if ($_.Exception.Message -match "permission denied|Authentication failed|403|401") {
                Throw-GitHubAuthError -RepoRoot $repoRoot -RemoteName $RemoteName -OriginalError $_.Exception.Message
            }

            throw
        }
    }

    if (-not $SkipPush -and ($needsPush -or $createdCommit)) {
        try {
            Invoke-Git -Arguments @("push", $RemoteName, $branch) | Out-Null
        }
        catch {
            if ($_.Exception.Message -match "permission denied|Authentication failed|403|401") {
                Throw-GitHubAuthError -RepoRoot $repoRoot -RemoteName $RemoteName -OriginalError $_.Exception.Message
            }

            throw
        }
    }

    $head = (Invoke-Git -Arguments @("rev-parse", "--short", "HEAD")).Trim()
    Write-Host ""
    Write-Host "Auto sync finished successfully."
    if ($createdCommit) {
        Write-Host "Commit hash: $head"
    }
    else {
        Write-Host "Commit hash: unchanged ($head)"
    }
}
finally {
    if ($DryRun -and $indexSnapshot -and (Test-Path $indexSnapshot)) {
        Restore-IndexSnapshot -GitDir $gitDir -SnapshotPath $indexSnapshot
        Remove-Item -LiteralPath $indexSnapshot -ErrorAction SilentlyContinue
    }
}
