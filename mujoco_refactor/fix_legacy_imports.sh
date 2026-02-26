#!/usr/bin/env bash
#
# Fix imports after moving legacy 13-dim config to archive/legacy/
# Updates utilities to use config_new (16-dim primary)
#

set -e
cd "$(dirname "$0")"

echo "========================================================================"
echo "Fixing legacy config imports → config_new"
echo "========================================================================"

# ============================================================================
# 1. Update utilities to use config_new (16-dim primary)
# ============================================================================

echo
echo "[1/9] Fixing show_bests.py..."
sed -i 's/^from config import/from config_new import/' show_bests.py

echo "[2/9] Fixing simulation_fast_new.py..."
sed -i 's/^from config import/from config_new import/' simulation_fast_new.py

echo "[3/9] Fixing terrain_test.py..."
sed -i 's/^from config import/from config_new import/' terrain_test.py

echo "[4/9] Fixing replay.py..."
sed -i 's/^from config import/from config_new import/' replay.py
sed -i 's/^import config$/import config_new as config/' replay.py

echo "[5/9] Fixing replay_best.py..."
sed -i 's/^from config import/from config_new import/' replay_best.py
sed -i 's/^from simulation_fast import/from simulation_fast_new import/' replay_best.py

echo "[6/9] Fixing replay_cmaes_state.py..."
sed -i 's/^from config import/from config_new import/' replay_cmaes_state.py
sed -i 's/^from simulation_fast import/from simulation_fast_new import/' replay_cmaes_state.py

echo "[7/9] Fixing visualize_rollout.py..."
sed -i 's/^from config import/from config_new import/' visualize_rollout.py

echo "[8/9] Fixing plot_torques.py..."
sed -i 's/^from config import/from config_new import/' plot_torques.py

# ============================================================================
# 2. Fix legacy-specific test to import from archive.legacy
# ============================================================================

echo "[9/9] Fixing test_flat_params_on_steps.py (legacy 13-dim)..."
# This file explicitly tests 13-dim flat params on step terrain
# Need to import from archive.legacy instead of config_new

# Add sys.path for archive imports at top of file (after imports section)
if ! grep -q "sys.path.insert.*archive" test_flat_params_on_steps.py; then
    sed -i '8a import sys\nsys.path.insert(0, "archive/legacy")' test_flat_params_on_steps.py
fi

# Update imports to use legacy modules (config already references config_step_13dim)
sed -i 's/^from config import/from config import/' test_flat_params_on_steps.py
sed -i 's/^from simulation_fast import/from simulation_fast import/' test_flat_params_on_steps.py
# Note: config_step_13dim import already correct (moved to archive/legacy with config_step.py→config_step_13dim.py rename)

echo
echo "========================================================================"
echo "✓ Legacy imports fixed!"
echo "========================================================================"
echo
echo "UPDATED FILES (config → config_new):"
echo "  - show_bests.py"
echo "  - simulation_fast_new.py"
echo "  - terrain_test.py"
echo "  - replay.py, replay_best.py, replay_cmaes_state.py"
echo "  - visualize_rollout.py"
echo "  - plot_torques.py"
echo
echo "LEGACY-SPECIFIC (imports from archive/legacy via sys.path):"
echo "  - test_flat_params_on_steps.py"
echo
echo "VERIFICATION:"
echo "  python show_bests.py results/20260225T122342_flat_10_30_50/"
echo "  grep 'from config' *.py | grep -v config_new | grep -v config_step"
echo
