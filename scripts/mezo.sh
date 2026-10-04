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
EPS="${EPS:-1e-3}"
LR="${LR:-1e-7}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
STEPS="${STEPS:-15000}"
EVAL_STEPS="${EVAL_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
TRAIN="${TRAIN:-2000}"
DEV="${DEV:-500}"
EVAL="${EVAL:-1000}"
LOAD_MODE="${LOAD_MODE:-bfloat16}"
MAX_TIME="${MAX_TIME:-0}"
SEED="${SEED:-0}"
REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-gemma}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-True}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%m%d-%H%M%S)}"

RANK="${RANK:-2}"
STEP_INTERVAL="${STEP_INTERVAL:-100}"
OPT="${OPT:-sgd}"
SPARSITY="${SPARSITY:-1.0}"

# TASK=SST2 | TASKS="SST2 RTE Copa"
TASKS="${TASKS:-${TASK:-SST2}}"
read -r -a TASK_LIST <<< "${TASKS}"
BASE_DEV="${DEV}"

export TRANSFORMERS_OFFLINE=1
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
MODEL_PATH="${MODEL_PATH:-$(resolve_local_model_path "${MODEL}")}"
TRAINER="mezo"
BASELINE="${BASELINE:-mezo}"

EXTRA_ARGS=""
if [ "$MODE" = "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" = "lora" ]; then
    EXTRA_ARGS="--lora"
fi

OVERWRITE_FLAG=""
if [ "${OVERWRITE_OUTPUT_DIR}" = "True" ] || [ "${OVERWRITE_OUTPUT_DIR}" = "true" ]; then
    OVERWRITE_FLAG="--overwrite_output_dir"
fi

LOAD_FLAG=""
if [ "$LOAD_MODE" = "bfloat16" ]; then
    LOAD_FLAG="--load_bfloat16"
elif [ "$LOAD_MODE" = "float16" ]; then
    LOAD_FLAG="--load_float16"
fi

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

    TAG="${TRAINER}-${MODE}-st${STEPS}-bs${BS}-lr${LR}-eps${EPS}-sd${SEED}-vi${STEP_INTERVAL}-r${RANK}-sp${SPARSITY}-opt${OPT}-${RUN_SUFFIX}"
    OUTPUT_DIR="$(result_output_dir "${RESULTS_ROOT}" "${BASELINE}" "${MODEL_NAME}" "${TASK}")"
    TASK_WANDB_NAME="${TASK}-${MODEL_NAME}-${TAG}"

    echo "Model: ${MODEL}"
    echo "Task: ${TASK}"
    echo "GPU(s): ${GPU}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Trainer: ${TRAINER}"
    echo "Optimizer: ${OPT}"
    echo "LR scheduler: ${LR_SCHEDULER_TYPE}"
    echo "Warmup steps: ${WARMUP_STEPS}"
    echo "Rank: ${RANK}"
    echo "Step interval: ${STEP_INTERVAL}"
    echo "Sparsity: ${SPARSITY}"
    echo "Overwrite output dir: ${OVERWRITE_OUTPUT_DIR}"

    WANDB_PROJECT="${WANDB_PROJECT_NAME}" WANDB_NAME="${TASK_WANDB_NAME}" CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" run.py \
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
        ${LOAD_FLAG} \
        --zo_optimizer "${OPT}" \
        --adam_eps 1e-6 \
        --max_time "${MAX_TIME}" \
        --sparsity "${SPARSITY}" \
        --learning_rate "${LR}" \
        --zo_eps "${EPS}" \
        --per_device_train_batch_size "${BS}" \
        --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --eval_strategy no \
        --save_strategy no \
        --step_interval "${STEP_INTERVAL}" \
        --rank_r "${RANK}" \
        --report_to "${REPORT_TO}" \
        ${OVERWRITE_FLAG} \
        ${TASK_ARGS} \
        ${EXTRA_ARGS} \
        --online_mode true \
        --max_stream_samples "${TRAIN}" \
        --stream_shuffle false \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 1 \
        --online_eval_before_update true \
        --replay_buffer_size 0 \
        --save_online_curve true \
        "$@"

    echo "Finished task: ${TASK}"
    echo "=========================================="
done
