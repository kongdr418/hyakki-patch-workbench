param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8787
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkbenchRoot = Split-Path -Parent $ScriptRoot
$BaseRoot = Split-Path -Parent $WorkbenchRoot
$OasRoot = Join-Path $BaseRoot "OnmyojiAutoScript-easy-install"
$Python = Join-Path $OasRoot "toolkit\python.exe"
$Collector = Join-Path $WorkbenchRoot "collector_app.py"
$DatasetRoot = Join-Path $WorkbenchRoot "datasets\patch"
$Url = "http://${HostName}:${Port}/"

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

$env:HYAKKI_OAS_ROOT = $OasRoot
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
