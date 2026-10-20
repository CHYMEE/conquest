# Waits for F, then calibrates the DP clip norm from block E and runs G (R2.1, DP-FedAvg).
param([int]$WaitPid)
$wd = "D:\conquest-milcom\revision"
Set-Location $wd
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 300 }
python make_grid_g.py >> "$wd\logs\queue.log" 2>&1
"$(Get-Date -Format s) starting G (DP-FedAvg)" | Out-File -Append "$wd\logs\queue.log"
python federated.py sweep --grid grids/G.json --workers 3 --threads 2 > "$wd\logs\G.out.log" 2> "$wd\logs\G.err.log"
"$(Get-Date -Format s) G finished (exit $LASTEXITCODE)" | Out-File -Append "$wd\logs\queue.log"
