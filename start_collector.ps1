$WorkbenchRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartScript = Join-Path $WorkbenchRoot "scripts\start_workbench.ps1"
& $StartScript @args
