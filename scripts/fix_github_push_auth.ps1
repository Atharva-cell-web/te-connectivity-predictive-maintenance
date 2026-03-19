[CmdletBinding()]
param(
    [string]$PrimaryRemoteName = "origin",
    [string]$PrimaryGitHubUsername = "atharvap2004",
    [string]$MirrorRemoteName = "vishh70",
    [string]$MirrorGitHubUsername = "Vishh70",
    [string]$GitHubRepo = "te-connectivity-predictive-maintenance",
    [switch]$SkipCredentialCleanup
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

$repoRoot = (Invoke-Git -Arguments @("rev-parse", "--show-toplevel")).Trim()
Set-Location $repoRoot

$remoteSpecs = @(
    New-RemoteSpec -Name $PrimaryRemoteName -GitHubUsername $PrimaryGitHubUsername -GitHubRepo $GitHubRepo
    New-RemoteSpec -Name $MirrorRemoteName -GitHubUsername $MirrorGitHubUsername -GitHubRepo $GitHubRepo
)
$configuredRemotes = foreach ($remoteSpec in $remoteSpecs) {
    Ensure-RepoGitHubRemote -RemoteSpec $remoteSpec
}
Clear-RepoCredentialUsername

$removedAccounts = @()
$removedTargets = @()
$allowedGitHubUsers = $remoteSpecs | ForEach-Object { $_.Username } | Sort-Object -Unique

if (-not $SkipCredentialCleanup) {
    $storedAccounts = & git credential-manager github list 2>$null | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ }
    foreach ($account in $storedAccounts) {
        if ($allowedGitHubUsers -contains $account) {
            continue
        }

        $logoutResult = Invoke-GitAllowFailure -Arguments @("credential-manager", "github", "logout", $account, "--no-ui")
        if ($logoutResult.ExitCode -eq 0) {
            $removedAccounts += $account
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

        if ($target -match 'git:https://(?<user>[^@]+)@github\.com$' -and $allowedGitHubUsers -notcontains $Matches.user) {
            Remove-WindowsCredential -Target $target
            $removedTargets += $target
        }
    }
}

Write-Host "Repository : $repoRoot"
Write-Host "Configured remotes:"
foreach ($configuredRemote in $configuredRemotes) {
    Write-Host "  $($configuredRemote.Name) -> $($configuredRemote.Url)"
}
Write-Host "Allowed GitHub users:"
foreach ($username in $allowedGitHubUsers) {
    Write-Host "  $username"
}

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
Write-Host "  Run 'git push origin main', 'git push vishh70 main', or '.\sync_project.cmd'."
Write-Host "  If GitHub prompts for login, sign in as 'atharvap2004' for origin and 'Vishh70' for vishh70."
