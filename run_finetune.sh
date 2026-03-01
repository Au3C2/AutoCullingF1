#!/usr/bin/env bash
# =============================================================================
# run_finetune.sh — Fine-tune all four backbone architectures sequentially.
#
# Usage:
#   bash run_finetune.sh [ARCH]
#
# If ARCH is provided (one of: resnet18 resnet50 resnext50 mobilenetv3),
# only that architecture is trained.  Otherwise all four are run in order.
#
# Key hyper-parameters are documented inline and can be overridden via
# environment variables, e.g.:
#   STEPS=2000 BATCH=16 bash run_finetune.sh resnet18
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable hyper-parameters (override via env vars)
# ---------------------------------------------------------------------------
TRAIN_CSV="${TRAIN_CSV:-dataset/train_info.csv}"
TEST_CSV="${TEST_CSV:-dataset/test_info.csv}"
IMG_SIZE="${IMG_SIZE:-224}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-4}"
VAL_FRAC="${VAL_FRAC:-0.1}"

STEPS="${STEPS:-20000}"          # Total training steps per run (upper bound; early stopping may halt sooner)
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"  # Validate every N steps
LR="${LR:-1e-3}"                # Head / full-model learning rate
BACKBONE_LR="${BACKBONE_LR:-1e-4}"  # Backbone learning rate (fine-tune)
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"  # Stop after N non-improving evals
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-1e-4}"
SEED="${SEED:-42}"

CKPT_BASE="${CKPT_BASE:-checkpoints}"

# ---------------------------------------------------------------------------
# Resolve the project root (directory containing this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the uv virtual environment if not already active
if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate 2>/dev/null || {
        echo "[ERROR] Could not activate .venv. Run 'uv sync' first." >&2
        exit 1
    }
fi

# ---------------------------------------------------------------------------
# Helper: train one architecture
# ---------------------------------------------------------------------------
run_arch() {
    local arch="$1"
    local ckpt_dir="${CKPT_BASE}/${arch}_finetune"

    echo ""
    echo "============================================================"
    echo "  Fine-tuning: ${arch}"
    echo "  Steps:       ${STEPS}  |  Eval every: ${EVAL_INTERVAL}"
    echo "  Batch:       ${BATCH}  |  LR: ${LR}  |  Backbone LR: ${BACKBONE_LR}"
    echo "  Checkpoint:  ${ckpt_dir}"
    echo "============================================================"

    uv run python3 -m auto_culling.train \
        --arch          "${arch}"                        \
        --train-csv     "${TRAIN_CSV}"                   \
        --test-csv      "${TEST_CSV}"                    \
        --img-size      "${IMG_SIZE}"                    \
        --batch-size    "${BATCH}"                       \
        --num-workers   "${WORKERS}"                     \
        --val-fraction  "${VAL_FRAC}"                    \
        --steps         "${STEPS}"                       \
        --eval-interval "${EVAL_INTERVAL}"               \
        --lr            "${LR}"                          \
        --backbone-lr   "${BACKBONE_LR}"                 \
        --weight-decay  "${WEIGHT_DECAY}"                \
        --warmup-ratio  "${WARMUP_RATIO}"                \
        --early-stopping-patience   "${EARLY_STOPPING_PATIENCE}"   \
        --early-stopping-min-delta  "${EARLY_STOPPING_MIN_DELTA}"  \
        --checkpoint-dir "${ckpt_dir}"                   \
        --seed          "${SEED}"                        \
        --finetune                                       \
        --pretrained

    echo "[OK] ${arch} finished → ${ckpt_dir}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_ARCHS=(resnet18 resnet50 resnext50 mobilenetv3)

if [ $# -ge 1 ]; then
    # Single-arch mode
    TARGET_ARCH="$1"
    VALID=0
    for a in "${ALL_ARCHS[@]}"; do
        [ "$a" = "$TARGET_ARCH" ] && VALID=1 && break
    done
    if [ "$VALID" -eq 0 ]; then
        echo "[ERROR] Unknown arch '${TARGET_ARCH}'. Choose from: ${ALL_ARCHS[*]}" >&2
        exit 1
    fi
    run_arch "$TARGET_ARCH"
else
    # All-arch mode
    for arch in "${ALL_ARCHS[@]}"; do
        run_arch "$arch"
    done
fi

echo ""
echo "All fine-tuning runs complete."
echo "Launch TensorBoard with:"
echo "  tensorboard --logdir ${CKPT_BASE}"
