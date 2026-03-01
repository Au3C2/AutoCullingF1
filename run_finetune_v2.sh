#!/usr/bin/env bash
# =============================================================================
# run_finetune_v2.sh — Anti-overfitting experiment (v2).
#
# Changes vs v1 (run_finetune.sh):
#   - weight_decay: 1e-4 → 5e-4
#   - label_smoothing: 0.0 → 0.1  (new arg in train.py)
#   - Dropout(0.3) before FC head  (in model.py)
#   - Unfreeze layer3 + layer4     (in model.py)
#   - Stronger augmentation        (RandAugment, GaussianBlur, RandomGrayscale)
#   - Checkpoint dir:  checkpoints/<arch>_finetune_v2
#
# Usage:
#   bash run_finetune_v2.sh [ARCH]
#
# ARCH: one of resnet18 resnet50 resnext50 mobilenetv3 (default: all)
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

STEPS="${STEPS:-20000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
LR="${LR:-1e-3}"
BACKBONE_LR="${BACKBONE_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"       # increased from 1e-4
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"  # new regulariser
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-1e-4}"
SEED="${SEED:-42}"

CKPT_BASE="${CKPT_BASE:-checkpoints}"

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "${VIRTUAL_ENV:-}" ]; then
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
    local ckpt_dir="${CKPT_BASE}/${arch}_finetune_v2"

    echo ""
    echo "============================================================"
    echo "  Fine-tuning v2: ${arch}"
    echo "  Steps:          ${STEPS}  |  Eval every: ${EVAL_INTERVAL}"
    echo "  Batch:          ${BATCH}  |  LR: ${LR}  |  Backbone LR: ${BACKBONE_LR}"
    echo "  Weight decay:   ${WEIGHT_DECAY}  |  Label smoothing: ${LABEL_SMOOTHING}"
    echo "  Checkpoint:     ${ckpt_dir}"
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
        --label-smoothing "${LABEL_SMOOTHING}"           \
        --warmup-ratio  "${WARMUP_RATIO}"                \
        --early-stopping-patience   "${EARLY_STOPPING_PATIENCE}"   \
        --early-stopping-min-delta  "${EARLY_STOPPING_MIN_DELTA}"  \
        --checkpoint-dir "${ckpt_dir}"                   \
        --seed          "${SEED}"                        \
        --finetune                                       \
        --pretrained

    echo "[OK] ${arch} v2 finished → ${ckpt_dir}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_ARCHS=(resnet18 resnet50 resnext50 mobilenetv3)

if [ $# -ge 1 ]; then
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
    for arch in "${ALL_ARCHS[@]}"; do
        run_arch "$arch"
    done
fi

echo ""
echo "All v2 fine-tuning runs complete."
echo "Launch TensorBoard with:"
echo "  tensorboard --logdir ${CKPT_BASE}"
