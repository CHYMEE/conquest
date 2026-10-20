# Waits for G, then runs H: variance and classical PSO at k in {10,20} (scenario 3 trade-off, R1.6 / R2.3).
param([int]$WaitPid)
$wd = "D:\conquest-milcom\revision"
Set-Location $wd
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 300 }
"$(Get-Date -Format s) starting H (variance/PSO at k=10,20)" | Out-File -Append "$wd\logs\queue.log"
python centralized.py sweep --grid grids/H.json --workers 3 --threads 2 > "$wd\logs\H.out.log" 2> "$wd\logs\H.err.log"
"$(Get-Date -Format s) H finished (exit $LASTEXITCODE)" | Out-File -Append "$wd\logs\queue.log"
