# Waits for the main queue (C+D, E, B2), then runs F: consensus mask + k in {10,20} (Reviewer 1, Comment 4).
param([int]$WaitPid)
$wd = "D:\conquest-milcom\revision"
Set-Location $wd
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 300 }
"$(Get-Date -Format s) starting F (consensus mask, k=10/20)" | Out-File -Append "$wd\logs\queue.log"
python centralized.py sweep --grid grids/F.json --workers 3 --threads 2 > "$wd\logs\F.out.log" 2> "$wd\logs\F.err.log"
"$(Get-Date -Format s) F finished (exit $LASTEXITCODE)" | Out-File -Append "$wd\logs\queue.log"
