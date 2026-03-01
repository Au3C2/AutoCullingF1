#!/usr/bin/env bash
# benchmark_onnx.sh
#
# 1. Export all four v2 checkpoints to ONNX (skips if already exported).
# 2. Run throughput benchmark for each model against a sample image directory.
# 3. Print a consolidated speed comparison table.
#
# Usage:
#   bash benchmark_onnx.sh [IMAGE_DIR]
#
# IMAGE_DIR defaults to dataset/cache (uses the training cache images as input).

set -euo pipefail

# ── Activate venv so uv + project packages are available ─────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.venv/bin/activate"

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR="${1:-${SCRIPT_DIR}/dataset/cache}"
ONNX_DIR="${SCRIPT_DIR}/onnx_models"
BATCH_SIZES="1 4 8 16 32"
IMG_SIZE=224

ARCHS=("resnet18" "resnet50" "resnext50" "mobilenetv3")

# ── Step 1: Export ONNX (skip if file already exists) ────────────────────────
echo "═══════════════════════════════════════════════════════"
echo " Step 1: ONNX Export"
echo "═══════════════════════════════════════════════════════"

EXPORT_NEEDED=0
for ARCH in "${ARCHS[@]}"; do
    if [[ ! -f "${ONNX_DIR}/${ARCH}.onnx" ]]; then
        EXPORT_NEEDED=1
        break
    fi
done

if [[ "${EXPORT_NEEDED}" -eq 1 ]]; then
    python "${SCRIPT_DIR}/export_onnx.py" \
        --output-dir "${ONNX_DIR}" \
        --img-size "${IMG_SIZE}"
else
    echo "All ONNX files already exist in ${ONNX_DIR}, skipping export."
fi

# ── Step 2: Benchmark each model ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Step 2: Throughput Benchmark"
echo " Image dir : ${IMAGE_DIR}"
echo " Batch sizes: ${BATCH_SIZES}"
echo "═══════════════════════════════════════════════════════"

# Temporary file to collect per-model results
RESULTS_FILE="$(mktemp /tmp/benchmark_results_XXXXXX.txt)"
trap 'rm -f "${RESULTS_FILE}"' EXIT

for ARCH in "${ARCHS[@]}"; do
    MODEL_PATH="${ONNX_DIR}/${ARCH}.onnx"
    if [[ ! -f "${MODEL_PATH}" ]]; then
        echo "[WARN] ONNX model not found, skipping: ${MODEL_PATH}"
        continue
    fi

    echo ""
    echo "─── ${ARCH} ──────────────────────────────────────────"
    python "${SCRIPT_DIR}/infer_onnx.py" \
        --model "${MODEL_PATH}" \
        --input-dir "${IMAGE_DIR}" \
        --img-size "${IMG_SIZE}" \
        --benchmark \
        --benchmark-batch-sizes ${BATCH_SIZES} \
        2>&1 | tee -a "${RESULTS_FILE}"
done

# ── Step 3: Summary table ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Summary: images/sec by model and batch size"
echo "═══════════════════════════════════════════════════════"
grep -E "(Model:|batch_size|[0-9]+\s+[0-9]+\.[0-9]+)" "${RESULTS_FILE}" || true

echo ""
echo "Benchmark complete."
