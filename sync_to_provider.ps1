# Sync LIRiAP_pack files to LiRiAP_provider/algorithms for release

$PACK_DIR = "LIRiAP_pack"
$PROVIDER_DIR = "LiRiAP_provider\algorithms"

Write-Host "Syncing LIRiAP_pack -> LiRiAP_provider..." -ForegroundColor Cyan

# Sync algorithm wrappers
Copy-Item "$PACK_DIR\*_algorithm.py" "$PROVIDER_DIR\" -Force

# Sync worker modules
Copy-Item "$PACK_DIR\*_worker.py" "$PROVIDER_DIR\" -Force

# Sync support files
Copy-Item "$PACK_DIR\numba_bootstrap.py" "$PROVIDER_DIR\" -Force
Copy-Item "$PACK_DIR\help_descriptions.py" "$PROVIDER_DIR\" -Force

Write-Host "Sync complete." -ForegroundColor Green

# Show updated files
Write-Host "`nUpdated files:" -ForegroundColor Yellow
Get-ChildItem "$PROVIDER_DIR\*_algorithm.py", "$PROVIDER_DIR\*_worker.py", "$PROVIDER_DIR\numba_bootstrap.py", "$PROVIDER_DIR\help_descriptions.py" | ForEach-Object { Write-Host "  $_" }
