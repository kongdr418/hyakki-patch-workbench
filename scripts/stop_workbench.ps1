param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8787
)

$ErrorActionPreference = "Continue"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkbenchRoot = Split-Path -Parent $ScriptRoot
$Url = "http://${HostName}:${Port}/"

function Stop-Tree {
    param([int]$ProcessId, [string]$Reason)
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $proc) { return }
    Write-Host "Stopping PID $($proc.Id) ($($proc.ProcessName)) - $Reason"
    # /T terminates the whole process tree (gunicorn workers etc.), /F forces.
    $output = & taskkill.exe /T /F /PID $proc.Id 2>&1
    if ($output) { Write-Host ($output | Out-String).TrimEnd() }
}

try {
    Invoke-RestMethod -Uri "${Url}api/train/stop" -Method Post -ContentType "application/json" -Body "{}" -TimeoutSec 5 | Out-Null
} catch {
}

Start-Sleep -Milliseconds 500

$workbenchRootEscaped = [regex]::Escape($WorkbenchRoot)
$workbenchProcesses = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match $workbenchRootEscaped -and
        ($_.CommandLine -match "train_patch_model\.py" -or $_.CommandLine -match "collector_app\.py")
    }

foreach ($proc in $workbenchProcesses) {
    Stop-Tree -ProcessId $proc.ProcessId -Reason "workbench process"
}

$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($conn in $listeners) {
    Stop-Tree -ProcessId $conn.OwningProcess -Reason "owns port $Port"
}

Start-Sleep -Milliseconds 800
$remaining = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Warning "Port $Port still in use after first pass, escalating."
    foreach ($conn in $remaining) {
        Stop-Tree -ProcessId $conn.OwningProcess -Reason "port $Port still bound"
    }
    Start-Sleep -Milliseconds 1500
    $remaining = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($remaining) {
        Write-Warning "Port $Port is still in use after escalation."
        foreach ($conn in $remaining) {
            Write-Host "Remaining PID: $($conn.OwningProcess)"
        }
        exit 1
    }
}

Write-Host "Hyakki patch workbench stopped."