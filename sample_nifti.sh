#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PROJECT_DIR:-}" ]]; then
    SCRIPT_DIR="$(cd "$PROJECT_DIR" && pwd)"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR}/config_sample_nifti.yaml" && \
          -f "${SLURM_SUBMIT_DIR}/super_res.py" ]]; then
        SCRIPT_DIR="$SLURM_SUBMIT_DIR"
    elif [[ -f "${SLURM_SUBMIT_DIR}/github/config_sample_nifti.yaml" && \
            -f "${SLURM_SUBMIT_DIR}/github/super_res.py" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/github"
    else
        echo "Cannot locate the standalone project under SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" >&2
        echo "Submit from the github directory, or set PROJECT_DIR=/path/to/github." >&2
        exit 1
    fi
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

INPUT_ROOT="${INPUT_ROOT:-/xxx/input_data}"  ## xxx needs to be changed to the truth path
CHECKPOINT="${CHECKPOINT:-/xxx/111.pt}"       ## xxx needs to be changed to the truth path
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
CONFIG="${CONFIG:-$SCRIPT_DIR/config_sample_nifti.yaml}"
DIRECTION="${DIRECTION:-axial}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for required in "$CONFIG" "$CHECKPOINT"; do
    if [[ ! -f "$required" ]]; then
        echo "Required file not found: $required" >&2
        exit 1
    fi
done
if [[ ! -d "$INPUT_ROOT" ]]; then
    echo "Input directory not found: $INPUT_ROOT" >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

MONITOR_PID=""
if [[ "${MONITOR_GPU:-1}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -l 5 > "$SCRIPT_DIR/nvidia02.log" 2>&1 &
    MONITOR_PID=$!
    trap '[[ -z "$MONITOR_PID" ]] || kill "$MONITOR_PID" 2>/dev/null || true' EXIT
fi

"$PYTHON_BIN" -u "$SCRIPT_DIR/super_res.py" \
    --config "$CONFIG" \
    --input-root "$INPUT_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --output-root "$OUTPUT_ROOT" \
    --direction "$DIRECTION"
