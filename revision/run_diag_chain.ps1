# Search-only QPSO diagnostics (no training): default budget (10 seeds), then 40x60 and 80x60 (5 seeds).
$wd = "D:\conquest-milcom\revision"
Set-Location $wd
python qpso_diagnostics.py run > "$wd\logs\diag.out.log" 2> "$wd\logs\diag.err.log"
python qpso_diagnostics.py run --particles 40 --iterations 60 --seeds 42,43,44,45,46 > "$wd\logs\diag_P40I60.out.log" 2> "$wd\logs\diag_P40I60.err.log"
python qpso_diagnostics.py run --particles 80 --iterations 60 --seeds 42,43,44,45,46 > "$wd\logs\diag_P80I60.out.log" 2> "$wd\logs\diag_P80I60.err.log"
"$(Get-Date -Format s) diagnostics chain finished" | Out-File -Append "$wd\logs\queue.log"
