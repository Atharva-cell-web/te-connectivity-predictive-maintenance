[CmdletBinding()]
param(
    [string]$RemoteName = "origin",
    [string]$GitHubUsername = "atharvap2004",
    [string]$GitHubOwner = "atharvap2004",
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

$repoRoot = (Invoke-Git -Arguments @("rev-parse", "--show-toplevel")).Trim()
Set-Location $repoRoot

$expectedRemoteUrl = "https://$GitHubUsername@github.com/$GitHubOwner/$GitHubRepo.git"
Invoke-Git -Arguments @("remote", "set-url", $RemoteName, $expectedRemoteUrl) | Out-Null
Invoke-Git -Arguments @("config", "--local", "credential.https://github.com.username", $GitHubUsername) | Out-Null

$removedAccounts = @()
$removedTargets = @()

if (-not $SkipCredentialCleanup) {
    $storedAccounts = & git credential-manager github list 2>$null | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ }
    foreach ($account in $storedAccounts) {
        if ($account -eq $GitHubUsername) {
            continue
        }

        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & git credential-manager github logout $account --no-ui 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $removedAccounts += $account
            }
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
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

        if ($target -match 'git:https://(?<user>[^@]+)@github\.com$' -and $Matches.user -ne $GitHubUsername) {
            Remove-WindowsCredential -Target $target
            $removedTargets += $target
        }
    }
}

Write-Host "Repository : $repoRoot"
Write-Host "Remote     : $expectedRemoteUrl"
Write-Host "GitHub user: $GitHubUsername"

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
Write-Host "  Run 'git push origin main' or '.\\sync_project.cmd'."
Write-Host "  If GitHub prompts for login, sign in as '$GitHubUsername'."
