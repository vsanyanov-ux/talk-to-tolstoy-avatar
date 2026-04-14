param (
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "stop", "status")]
    $Action
)

$backendPort = 8001
$frontendPort = 5199
$projectPath = "c:\Users\vanya\Antigravity Projects\Apps\Talk to Tolstoy Avatar"
$backendPath = "$projectPath\backend"
$frontendPath = "$projectPath\frontend"

function Get-ProcessByPort($port) {
    $result = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($result) {
        return Get-Process -Id $result.OwningProcess -ErrorAction SilentlyContinue
    }
    return $null
}

function Show-Status {
    Write-Host "--- Tolstoy Avatar Status ---"
    $bp = Get-ProcessByPort $backendPort
    $fp = Get-ProcessByPort $frontendPort
    
    $bStatus = if ($bp) { "[ONLINE] (Port $backendPort)" } else { "[OFFLINE]" }
    $fStatus = if ($fp) { "[ONLINE] (Port $frontendPort)" } else { "[OFFLINE]" }
    
    Write-Host "Backend         : $bStatus"
    Write-Host "Avatar Frontend : $fStatus"
    Write-Host ""
}

if ($Action -eq "stop" -or $Action -eq "start") {
    Write-Host "--- Stopping Processes on Ports $backendPort, $frontendPort ---"
    foreach ($port in @($backendPort, $frontendPort)) {
        $p = Get-ProcessByPort $port
        if ($p) {
            Write-Host "Killing process $($p.Id) on port $port..."
            Stop-Process -Id $p.Id -Force
        }
    }
    Write-Host "Cleanup complete."
}

if ($Action -eq "start") {
    Write-Host "--- Starting Tolstoy Avatar ---"
    
    # Start Backend
    Write-Host "Launching Backend (Port $backendPort)..."
    Start-Process "python" -ArgumentList "server.py" -WorkingDirectory $backendPath -WindowStyle Hidden -RedirectStandardOutput "$projectPath\backend.log" -RedirectStandardError "$projectPath\backend_err.log"
    
    # Start Frontend (Use cmd /c for npm)
    Write-Host "Launching Frontend (Port $frontendPort)..."
    Start-Process "cmd.exe" -ArgumentList "/c npm run dev" -WorkingDirectory $frontendPath -WindowStyle Hidden -RedirectStandardOutput "$projectPath\frontend.log" -RedirectStandardError "$projectPath\frontend_err.log"
    
    Write-Host "Services starting. Use './manage.ps1 status' to check."
}

if ($Action -eq "status") {
    Show-Status
}
