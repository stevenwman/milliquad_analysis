#!/usr/bin/env bash
#
# Fix imports after moving multi-terrain files to experiments/ and renaming config_step files.
# Run this AFTER manually moving the files.
#
# Expected file moves (done manually):
#   config_multi_terrain.py → experiments/config_multi_terrain.py
#   optimizer_multi_terrain.py → experiments/optimizer_multi_terrain.py
#   show_bests_multi_terrain.py → experiments/show_bests_multi_terrain.py
#   MULTI_TERRAIN_OPTIMIZER_SUMMARY.md → experiments/MULTI_TERRAIN_OPTIMIZER_SUMMARY.md
#   config_step.py → config_step_13dim.py
#   config_step_new.py → config_step.py
#

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================================================"
echo "Fixing imports after file reorganization"
echo "========================================================================"

# ============================================================================
# 1. Fix imports for multi-terrain files moved to experiments/
# ============================================================================
# Note: Files IN experiments/ don't need changes (import from same dir)
# We're only fixing files OUTSIDE experiments/ that import multi-terrain modules

echo
echo "[1/7] Skipping files in experiments/ (no changes needed)..."
echo "  - test_batch_size.py (in experiments/, imports from same dir)"
echo "  - verify_reference_velocities.py (in experiments/, imports from same dir)"
echo "  - test_terrain_cost_balance.py (in experiments/, imports from same dir)"

# ============================================================================
# 2. Fix imports for config_step rename (config_step_new → config_step)
# ============================================================================

echo "[4/7] Fixing optimizer_step.py (2 imports)..."
sed -i 's/^from config_step_new import/from config_step import/' optimizer_step.py
sed -i 's/from config_step_new import sim_params_from_point/from config_step import sim_params_from_point/' optimizer_step.py

# ============================================================================
# 3. Fix imports for old 13-dim config_step (config_step → config_step_13dim)
# ============================================================================

echo "[5/7] Fixing test_flat_params_on_steps.py..."
sed -i 's/^from config_step import/from config_step_13dim import/' test_flat_params_on_steps.py

echo "[6/7] Fixing archive/old files/bodyflip_analysis/compare_bodyflip_steps.py..."
if [ -f "archive/old files/bodyflip_analysis/compare_bodyflip_steps.py" ]; then
    sed -i 's/^from config_step import/from config_step_13dim import/' "archive/old files/bodyflip_analysis/compare_bodyflip_steps.py"
else
    echo "  (File not found, skipping)"
fi

# ============================================================================
# 4. No changes needed for these files (auto-fixed by rename):
# ============================================================================

echo "[7/7] Verifying auto-fixed files (no changes needed)..."
echo "  - show_bests_step.py (imports config_step, which is now the 16-dim version)"
echo "  - test_16dim_params_on_steps.py (imports config_step, which is now the 16-dim version)"

echo
echo "========================================================================"
echo "✓ All imports fixed!"
echo "========================================================================"
echo
echo "VERIFICATION CHECKS:"
echo "1. Multi-terrain files in experiments/:"
echo "   ls -1 experiments/"
echo
echo "2. Config_step files renamed:"
echo "   ls -1 config_step*.py"
echo
echo "3. Grep for any remaining old imports:"
echo "   grep -r 'from config_multi_terrain import' *.py 2>/dev/null | grep -v experiments || echo '  (none found)'"
echo "   grep -r 'from optimizer_multi_terrain import' *.py 2>/dev/null | grep -v experiments || echo '  (none found)'"
echo "   grep -r 'from config_step_new import' *.py 2>/dev/null || echo '  (none found)'"
echo
