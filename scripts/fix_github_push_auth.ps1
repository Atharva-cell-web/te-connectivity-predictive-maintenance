[CmdletBinding()]
param(
    [string]$RemoteName = "origin",
    [string]$GitHubOwner = "Atharva-cell-web",
    [string]$GitHubRepo = "te-connectivity-predictive-maintenance",
    [switch]$ResetGitHubCredentialsOnly
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

function Remove-WindowsCredential {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Target
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & cmdkey "/delete:$Target" 2>&1 | ForEach-Object { $_.ToString() }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0 -and ($output -join " ") -notmatch "cannot find") {
        throw "cmdkey /delete:$Target failed:`n$output"
    }
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

$PrimaryRemoteName = $RemoteName
$PrimaryGitHubOwner = $GitHubOwner
$MirrorRemoteName = "vishnu"
$MirrorGitHubOwner = "Vishh70"

$repoRoot = (Invoke-Git -Arguments @("rev-parse", "--show-toplevel")).Trim()
Set-Location $repoRoot

Normalize-LegacyMirrorRemote -PreferredRemoteName $MirrorRemoteName -LegacyRemoteName "vishh70"

$remoteSpecs = @(
    New-RemoteSpec -Name $PrimaryRemoteName -GitHubOwner $PrimaryGitHubOwner -GitHubRepo $GitHubRepo
)

if ($MirrorRemoteName -ne $PrimaryRemoteName) {
    $remoteSpecs += New-RemoteSpec -Name $MirrorRemoteName -GitHubOwner $MirrorGitHubOwner -GitHubRepo $GitHubRepo
}

$configuredRemotes = foreach ($remoteSpec in $remoteSpecs) {
    Ensure-GitHubRemote -RemoteSpec $remoteSpec
}

Clear-RepoCredentialUsername

$removedAccounts = @()
$removedTargets = @()

if (-not $ResetGitHubCredentialsOnly) {
    $storedAccounts = & git credential-manager github list 2>$null | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ }
    foreach ($account in $storedAccounts) {
        $logoutResult = Invoke-GitAllowFailure -Arguments @("credential-manager", "github", "logout", $account, "--no-ui")
        if ($logoutResult.ExitCode -eq 0) {
            $removedAccounts += $account
        }
    }
}

$cmdkeyOutput = & cmdkey /list 2>&1 | ForEach-Object { $_.ToString() }
$targets = foreach ($line in $cmdkeyOutput) {
    if ($line -match 'Target:\s+(?<target>LegacyGeneric:target=git:https://[^ ]+)') {
        $matches.target
    }
}

foreach ($target in $targets | Sort-Object -Unique) {
    if ($target -match 'git:https://github\.com$') {
        Remove-WindowsCredential -Target $target
        $removedTargets += $target
        continue
    }

    if ($target -match 'git:https://[^@]+@github\.com$') {
        Remove-WindowsCredential -Target $target
        $removedTargets += $target
    }
}

Write-Host "Repository : $repoRoot"
Write-Host "Configured remotes:"
foreach ($configuredRemote in $configuredRemotes) {
    Write-Host "  $($configuredRemote.Name) -> $($configuredRemote.Url)"
}
Write-Host "Mode       : dual remote auto sync"

if ($removedAccounts.Count -gt 0) {
    Write-Host "Removed GCM accounts:"
    foreach ($account in $removedAccounts) {
        Write-Host "  $account"
    }
}

if ($removedTargets.Count -gt 0) {
    Write-Host "Removed Windows credentials:"
    foreach ($target in $removedTargets) {
        Write-Host "  $target"
    }
}

Write-Host ""
Write-Host "Next step:"
Write-Host "  Run '.\sync_project.cmd' or 'git autosync'."
Write-Host "  The current branch will be pushed to both origin and vishnu."
Write-Host "  Sign in with a GitHub account that can push to both repos if Git asks again."

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

function Remove-WindowsCredential {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Target
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & cmdkey "/delete:$Target" 2>&1 | ForEach-Object { $_.ToString() }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0 -and ($output -join " ") -notmatch "cannot find") {
        throw "cmdkey /delete:$Target failed:`n$output"
    }
}

function Clear-RepoCredentialUsername {
    $result = Invoke-GitAllowFailure -Arguments @("config", "--local", "--unset-all", "credential.https://github.com.username")
    $outputText = ($result.Output -join " ").Trim()
    if ($result.ExitCode -ne 0 -and $outputText -and $outputText -notmatch "No such section or key") {
        throw "git config --local --unset-all credential.https://github.com.username failed:`n$($result.Output)"
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

function Remove-MirrorRemoteIfPresent {
    $remoteResult = Invoke-GitAllowFailure -Arguments @("remote")
    if ($remoteResult.ExitCode -ne 0) {
        return
    }

    $remotes = $remoteResult.Output | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ }
    if ($remotes -contains "vishh70") {
        Invoke-Git -Arguments @("remote", "remove", "vishh70") | Out-Null
    }
}

$repoRoot = (Invoke-Git -Arguments @("rev-parse", "--show-toplevel")).Trim()
Set-Location $repoRoot

$remoteUrl = Ensure-SharedRemote -RemoteName $RemoteName -GitHubOwner $GitHubOwner -GitHubRepo $GitHubRepo
Clear-RepoCredentialUsername
Remove-MirrorRemoteIfPresent

$removedAccounts = @()
$removedTargets = @()

if (-not $ResetGitHubCredentialsOnly) {
    $storedAccounts = & git credential-manager github list 2>$null | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ }
    foreach ($account in $storedAccounts) {
        $logoutResult = Invoke-GitAllowFailure -Arguments @("credential-manager", "github", "logout", $account, "--no-ui")
        if ($logoutResult.ExitCode -eq 0) {
            $removedAccounts += $account
        }
    }
}

$cmdkeyOutput = & cmdkey /list 2>&1 | ForEach-Object { $_.ToString() }
$targets = foreach ($line in $cmdkeyOutput) {
    if ($line -match 'Target:\s+(?<target>LegacyGeneric:target=git:https://[^ ]+)') {
        $matches.target
    }
}

foreach ($target in $targets | Sort-Object -Unique) {
    if ($target -match 'git:https://github\.com$') {
        Remove-WindowsCredential -Target $target
        $removedTargets += $target
        continue
    }

    if ($target -match 'git:https://[^@]+@github\.com$') {
        Remove-WindowsCredential -Target $target
        $removedTargets += $target
    }
}

Write-Host "Repository : $repoRoot"
Write-Host "Remote     : $RemoteName -> $remoteUrl"
Write-Host "Mode       : single shared GitHub repo"

if ($removedAccounts.Count -gt 0) {
    Write-Host "Removed GCM accounts:"
    foreach ($account in $removedAccounts) {
        Write-Host "  $account"
    }
}

if ($removedTargets.Count -gt 0) {
    Write-Host "Removed Windows credentials:"
    foreach ($target in $removedTargets) {
        Write-Host "  $target"
    }
}

Write-Host ""
Write-Host "Next step:"
Write-Host "  Run 'git push origin main' or '.\sync_project.cmd'."
Write-Host "  Sign in with any GitHub account that has collaborator access to the shared repo."
