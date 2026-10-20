# Waits for the running A+B sweep, then runs C+D, E and B2 in sequence (all resumable).
param([int]$WaitPid)
$wd = "D:\conquest-milcom\revision"
Set-Location $wd
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 300 }
"$(Get-Date -Format s) A+B finished; starting C+D" | Out-File -Append "$wd\logs\queue.log"
python centralized.py sweep --grid grids/CD.json --workers 3 --threads 2 > "$wd\logs\CD.out.log" 2> "$wd\logs\CD.err.log"
"$(Get-Date -Format s) C+D finished (exit $LASTEXITCODE); starting E" | Out-File -Append "$wd\logs\queue.log"
python federated.py sweep --grid grids/E.json --workers 3 --threads 2 > "$wd\logs\E.out.log" 2> "$wd\logs\E.err.log"
"$(Get-Date -Format s) E finished (exit $LASTEXITCODE); starting B2" | Out-File -Append "$wd\logs\queue.log"
python centralized.py sweep --grid grids/B2.json --workers 3 --threads 2 > "$wd\logs\B2.out.log" 2> "$wd\logs\B2.err.log"
"$(Get-Date -Format s) B2 finished (exit $LASTEXITCODE)" | Out-File -Append "$wd\logs\queue.log"
