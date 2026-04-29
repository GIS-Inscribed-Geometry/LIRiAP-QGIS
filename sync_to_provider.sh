#!/bin/bash
# Sync LIRiAP_pack files to LiRiAP_provider/algorithms for release

# Source and target directories
PACK_DIR="LIRiAP_pack"
PROVIDER_DIR="LiRiAP_provider/algorithms"

echo "Syncing LIRiAP_pack -> LiRiAP_provider..."

# Sync algorithm wrappers
cp "$PACK_DIR"/*_algorithm.py "$PROVIDER_DIR/"

# Sync worker modules  
cp "$PACK_DIR"/*_worker.py "$PROVIDER_DIR/"

# Sync support files
cp "$PACK_DIR"/numba_bootstrap.py "$PROVIDER_DIR/"
cp "$PACK_DIR"/help_descriptions.py "$PROVIDER_DIR/"

echo "Sync complete."

# Show what changed
echo ""
echo "Updated files:"
ls -la "$PROVIDER_DIR"/*_algorithm.py "$PROVIDER_DIR"/*_worker.py "$PROVIDER_DIR"/numba_bootstrap.py "$PROVIDER_DIR"/help_descriptions.py 2>/dev/null | awk '{print $NF}'
