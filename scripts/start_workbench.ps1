param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8787,
    [string]$OasRoot = ""
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkbenchRoot = Split-Path -Parent $ScriptRoot
$BaseRoot = Split-Path -Parent $WorkbenchRoot
$LocalConfig = Join-Path $WorkbenchRoot "config.local.json"
$Collector = Join-Path $WorkbenchRoot "collector_app.py"
$DatasetRoot = Join-Path $WorkbenchRoot "datasets\patch"
$Url = "http://${HostName}:${Port}/"

function Test-OasRoot {
    param([string]$Path)
    if (-not $Path) {
        return $false
    }
    return (
        (Test-Path -LiteralPath (Join-Path $Path "toolkit\python.exe")) -and
        (Test-Path -LiteralPath (Join-Path $Path "module\config\config.py")) -and
        (Test-Path -LiteralPath (Join-Path $Path "tasks\Hyakkiyakou"))
    )
}

function Resolve-OasRoot {
    $candidates = @()
    if ($OasRoot) {
        $candidates += $OasRoot
    }
    if ($env:HYAKKI_OAS_ROOT) {
        $candidates += $env:HYAKKI_OAS_ROOT
    }
    if (Test-Path -LiteralPath $LocalConfig) {
        try {
            $local = Get-Content -LiteralPath $LocalConfig -Raw | ConvertFrom-Json
            if ($local.oas_root) {
                $candidates += [string]$local.oas_root
            }
        } catch {
            Write-Warning "Cannot read config.local.json: $($_.Exception.Message)"
        }
    }
    $candidates += (Join-Path $BaseRoot "OnmyojiAutoScript-easy-install")
    Get-ChildItem -LiteralPath $BaseRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $candidates += $_.FullName
    }

    $seen = @{}
    foreach ($candidate in $candidates) {
        if (-not $candidate) {
            continue
        }
        $resolvedPath = Resolve-Path -LiteralPath $candidate -ErrorAction SilentlyContinue
        if ($resolvedPath) {
            $resolved = $resolvedPath.ProviderPath
        } else {
            $resolved = [System.IO.Path]::GetFullPath($candidate)
        }
        $key = $resolved.ToLowerInvariant()
        if ($seen.ContainsKey($key)) {
            continue
        }
        $seen[$key] = $true
        if (Test-OasRoot -Path $resolved) {
            return $resolved
        }
    }

    throw @"
Cannot find an OAS folder.
The folder name does not matter, but it must contain:
  toolkit\python.exe
  module\config\config.py
  tasks\Hyakkiyakou\

Pass the OAS path explicitly:
  .\start_collector.ps1 -OasRoot "D:\path\to\your-oas-folder"

Or create config.local.json:
  { "oas_root": "D:\\path\\to\\your-oas-folder" }
"@
}

$ResolvedOasRoot = Resolve-OasRoot
$Python = Join-Path $ResolvedOasRoot "toolkit\python.exe"

function Test-ServiceReady {
    param([string]$StateUrl)
    try {
        Invoke-RestMethod -Uri $StateUrl -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Cannot find OAS Python: $Python"
}
if (-not (Test-Path -LiteralPath $Collector)) {
    throw "Cannot find collector app: $Collector"
}

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Hyakki patch workbench is already running."
    Write-Host $Url
    exit 0
}

$env:HYAKKI_OAS_ROOT = $ResolvedOasRoot
$args = @(
    $Collector,
    "--host", $HostName,
    "--port", $Port,
    "--root", $DatasetRoot
)

$process = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $WorkbenchRoot -WindowStyle Hidden -PassThru

$stateUrl = "${Url}api/state"
for ($i = 0; $i -lt 30; $i++) {
    if (Test-ServiceReady -StateUrl $stateUrl) {
        Write-Host "Hyakki patch workbench started."
        Write-Host "PID: $($process.Id)"
        Write-Host $Url
        exit 0
    }
    Start-Sleep -Milliseconds 500
}

Write-Warning "Started process $($process.Id), but the service did not respond within 15 seconds."
Write-Host $Url
