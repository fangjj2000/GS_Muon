#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/online_env.sh"
RESULTS_ROOT="${RESULTS_ROOT:-${LLM_DIR}/results}"

MODEL="${MODEL:-facebook/opt-13b}"
MODE="${MODE:-ft}"
GPU="${GPU:-1}"
BS="${BS:-1}"
LR="${LR:-5e-7}"
EPS="${EPS:-1e-3}"
SEED="${SEED:-0}"
TRAIN="${TRAIN:-2000}"
DEV="${DEV:-500}"
EVAL="${EVAL:-1000}"
STEPS="${STEPS:-10000}"
EVAL_STEPS="${EVAL_STEPS:-5000}"
WARMUP_STEP="${WARMUP_STEP:-0}"
DECAY_STEP="${DECAY_STEP:-0}"
ZO_LR_SCHEDULER_TYPE="${ZO_LR_SCHEDULER_TYPE:-constant}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
HESSIAN_SMOOTH_TYPE="${HESSIAN_SMOOTH_TYPE:-constant0}"
LOAD_MODE="${LOAD_MODE:-bfloat16}"
MAX_TIME="${MAX_TIME:-0}"
NO_ALIGN="${NO_ALIGN:-}"
REPORT_TO="${REPORT_TO:-none}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%m%d-%H%M%S)}"

# TASK=SST2 | TASKS="SST2 RTE Copa"
TASKS="${TASKS:-${TASK:-SST2}}"
read -r -a TASK_LIST <<< "${TASKS}"
BASE_DEV="${DEV}"

MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
TRAINER="hizoo"
BASELINE="${BASELINE:-hizoo}"
export TRANSFORMERS_OFFLINE=1
MODEL_PATH="${MODEL_PATH:-$(resolve_local_model_path "${MODEL}")}"

EXTRA_ARGS=""
if [ "$MODE" = "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" = "lora" ]; then
    EXTRA_ARGS="--lora"
fi

LOAD_FLAG=""
if [ "$LOAD_MODE" = "bfloat16" ]; then
    LOAD_FLAG="--load_bfloat16"
elif [ "$LOAD_MODE" = "float16" ]; then
    LOAD_FLAG="--load_float16"
fi

TAG="${TRAINER}-${MODE}-st${STEPS}-bs${BS}-lr${LR}-eps${EPS}-sd${SEED}-${HESSIAN_SMOOTH_TYPE}${NO_ALIGN:+-$NO_ALIGN}-${RUN_SUFFIX}"
mkdir -p "${RESULTS_ROOT}/${BASELINE}/${MODEL_NAME}"
cd "${LLM_DIR}"

echo "Tasks to run: ${TASK_LIST[*]}"

for TASK in "${TASK_LIST[@]}"; do
    echo "=========================================="
    echo "Starting task: ${TASK}"
    echo "=========================================="

    DEV="${BASE_DEV}"
    TASK_ARGS="--train_as_classification"
    case "$TASK" in
        CB)
            DEV=100
            ;;
        Copa)
            DEV=100
            TASK_ARGS="--train_as_classification False"
            ;;
        ReCoRD|DROP|SQuAD)
            TASK_ARGS="--train_as_classification False"
            ;;
    esac

    OUTPUT_DIR="$(result_output_dir "${RESULTS_ROOT}" "${BASELINE}" "${MODEL_NAME}" "${TASK}")"

    echo "TAG: ${TAG}"
    echo "Task: ${TASK}"
    echo "BS: ${BS}"
    echo "LR: ${LR}"
    echo "EPS: ${EPS}"
    echo "SEED: ${SEED}"
    echo "STREAM SAMPLES: ${TRAIN}"
    echo "MODE: ${MODE}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Extra args: ${EXTRA_ARGS} ${TASK_ARGS}"

    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" run.py \
        --model_name "${MODEL}" \
        --model_path "${MODEL_PATH}" \
        --task_name "${TASK}" \
        --output_dir "${OUTPUT_DIR}" \
        --tag "${TAG}" \
        --train_set_seed "${SEED}" \
        --num_train "${TRAIN}" \
        --logging_steps 100 \
        --max_steps "${STEPS}" \
        --trainer "${TRAINER}" \
        --max_time "${MAX_TIME}" \
        ${LOAD_FLAG} \
        --learning_rate "${LR}" \
        --zo_eps "${EPS}" \
        --per_device_train_batch_size "${BS}" \
        --lr_scheduler_type constant \
        --eval_strategy no \
        --save_strategy no \
        --warmup_step "${WARMUP_STEP}" \
        --decay_step "${DECAY_STEP}" \
        --zo_lr_scheduler_type "${ZO_LR_SCHEDULER_TYPE}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --hessian_smooth_type "${HESSIAN_SMOOTH_TYPE}" \
        ${EXTRA_ARGS} \
        ${TASK_ARGS} \
        --online_mode true \
        --max_stream_samples "${TRAIN}" \
        --stream_shuffle false \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 1 \
        --online_eval_before_update true \
        --replay_buffer_size 0 \
        --save_online_curve true \
        --report_to "${REPORT_TO}" \
        "$@"

    echo "Finished task: ${TASK}"
    echo "=========================================="
done
