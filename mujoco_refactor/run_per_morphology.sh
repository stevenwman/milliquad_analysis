#!/usr/bin/env bash
# Run independent optimizations split by morphology, frequency, and combined.
# Usage: bash run_per_morphology.sh [n_calls] [--loop-morphs | --loop-freqs | --loop-combined]
set -euo pipefail
cd "$(dirname "$0")"

N=${1:-600}
MODE=${2:-all}  # all, --loop-morphs, --loop-freqs, --loop-combined

if [[ "$MODE" == "all" || "$MODE" == "--loop-morphs" ]]; then
    echo "=== Per-morphology optimization: $N evals each ==="
    for scene in scene1 scene2 scene4 scene_wheel; do
        echo ""
        echo ">>> Starting $scene ($N evals) ..."
        uv run python optimizer.py --scenes "$scene" --n-calls "$N" --suffix "solo_${scene}"
        echo ">>> Finished $scene"
    done
fi

if [[ "$MODE" == "all" || "$MODE" == "--loop-freqs" ]]; then
    echo ""
    echo "=== Per-frequency optimization: $N evals each ==="
    for freq in 10 30 50; do
        echo ""
        echo ">>> Starting f${freq}Hz ($N evals) ..."
        uv run python optimizer.py --freqs "$freq" --n-calls "$N" --suffix "solo_f${freq}"
        echo ">>> Finished f${freq}Hz"
    done
fi

if [[ "$MODE" == "all" || "$MODE" == "--loop-combined" ]]; then
    echo ""
    echo "=== Combined optimization (all morphologies + frequencies): $N evals ==="
    uv run python optimizer.py --n-calls "$N" --suffix "combined"
    echo ">>> Finished combined"
fi

echo ""
echo "=== Done. Run: uv run python compare_morphology_params.py ==="
