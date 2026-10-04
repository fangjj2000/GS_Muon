#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/online_env.sh"
RESULTS_ROOT="${RESULTS_ROOT:-${LLM_DIR}/results}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
MODE="${MODE:-ft}"
GPU="${GPU:-0}"
BS="${BS:-1}"
LR="${LR:-}"
EPOCH="${EPOCH:-1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
EVAL_STEPS="${EVAL_STEPS:-200}"
TRAIN="${TRAIN:-2000}"
DEV="${DEV:-500}"
EVAL="${EVAL:-1000}"
LOAD_MODE="${LOAD_MODE:-bfloat16}"
SEED="${SEED:-0}"
REPORT_TO="${REPORT_TO:-none}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-True}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%m%d-%H%M%S)}"
MUON_BETA="${MUON_BETA:-0.95}"
MUON_NS_STEPS="${MUON_NS_STEPS:-5}"
GG_NS_ETA_G="${GG_NS_ETA_G:-0.50}"
GG_NS_PRE_STEPS="${GG_NS_PRE_STEPS:-3}"

# TASK=SST2 | TASKS="SST2 RTE Copa"
TASKS="${TASKS:-${TASK:-SST2}}"
read -r -a TASK_LIST <<< "${TASKS}"
BASE_DEV="${DEV}"
BASE_BS="${BS}"

MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
TRAINER="regular"
MODEL_PATH="${MODEL_PATH:-$(resolve_local_model_path "${MODEL}")}"
USE_MUON="${USE_MUON:-false}"
DRIFT_AWARE="${DRIFT_AWARE:-false}"
BASELINE="${BASELINE:-}"
if [ -z "${BASELINE}" ]; then
    if [ "${USE_MUON,,}" = "true" ]; then
        if [ "${DRIFT_AWARE,,}" = "true" ]; then
            BASELINE="ft_muon_da"
        else
            BASELINE="ft_muon"
        fi
    else
        BASELINE="ft"
    fi
fi
if [ -z "${LR}" ]; then
    if [ "${USE_MUON,,}" = "true" ]; then
        LR="1e-4"
    else
        LR="1e-5"
    fi
fi

EXTRA_ARGS=""
if [ "${MODE}" = "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "${MODE}" = "lora" ]; then
    EXTRA_ARGS="--lora"
fi

OVERWRITE_FLAG=""
if [ "${OVERWRITE_OUTPUT_DIR}" = "True" ] || [ "${OVERWRITE_OUTPUT_DIR}" = "true" ]; then
    OVERWRITE_FLAG="--overwrite_output_dir"
fi

LOAD_FLAG=""
if [ "${LOAD_MODE}" = "bfloat16" ]; then
    LOAD_FLAG="--load_bfloat16"
elif [ "${LOAD_MODE}" = "float16" ]; then
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
    BS="${BASE_BS}"
    TASK_ARGS="--train_as_classification"
    case "${TASK}" in
        CB)
            DEV=100
            ;;
        Copa)
            DEV=100
            TASK_ARGS="--train_as_classification False"
            ;;
        MultiRC)
            GA=1
            echo "Gradient accumulation: ${GA}"
            TASK_ARGS="--gradient_accumulation_steps ${GA}"
            ;;
        ReCoRD)
            GA=1
            echo "Gradient accumulation: ${GA}"
            TASK_ARGS="--gradient_accumulation_steps ${GA} --train_as_classification False"
            ;;
        DROP)
            GA=1
            echo "Gradient accumulation: ${GA}"
            TASK_ARGS="--gradient_accumulation_steps ${GA} --train_as_classification False"
            ;;
        SQuAD)
            TASK_ARGS="--train_as_classification False"
            ;;
    esac

    BS=1

    TAG="${TRAINER}-${MODE}-ep${EPOCH}-bs${BS}-lr${LR}-sd${SEED}-${RUN_SUFFIX}"
    OUTPUT_DIR="$(result_output_dir "${RESULTS_ROOT}" "${BASELINE}" "${MODEL_NAME}" "${TASK}")"

    echo "Model: ${MODEL}"
    echo "LR scheduler: ${LR_SCHEDULER_TYPE}"
    echo "Task: ${TASK}"
    echo "GPU(s): ${GPU}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Trainer: ${TRAINER}"
    echo "Epochs: ${EPOCH}"
    echo "Batch size: ${BS}"
    echo "LR: ${LR}"
    echo "Seed: ${SEED}"
    echo "Mode: ${MODE}"
    echo "Load mode: ${LOAD_MODE}"
    echo "Overwrite output dir: ${OVERWRITE_OUTPUT_DIR}"
    echo "Extra args: ${EXTRA_ARGS} ${TASK_ARGS}"

    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" run.py \
        --model_name "${MODEL}" \
        --model_path "${MODEL_PATH}" \
        --task_name "${TASK}" \
        --output_dir "${OUTPUT_DIR}" \
        --tag "${TAG}" \
        --train_set_seed "${SEED}" \
        --num_train "${TRAIN}" \
        --logging_steps 10 \
        --trainer "${TRAINER}" \
        ${LOAD_FLAG} \
        --learning_rate "${LR}" \
        --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
        --num_train_epochs "${EPOCH}" \
        --per_device_train_batch_size "${BS}" \
        --eval_strategy no \
        --save_strategy no \
        --report_to "${REPORT_TO}" \
        ${OVERWRITE_FLAG} \
        ${TASK_ARGS} \
        ${EXTRA_ARGS} \
        --online_mode true \
        --adaptation_method ft \
        --max_stream_samples "${TRAIN}" \
        --stream_shuffle false \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 1 \
        --online_eval_before_update true \
        --replay_buffer_size 0 \
        --save_online_curve true \
        --use_muon "${USE_MUON}" \
        --muon_beta "${MUON_BETA}" \
        --muon_ns_steps "${MUON_NS_STEPS}" \
        --drift_aware "${DRIFT_AWARE}" \
        --gg_ns_eta_g "${GG_NS_ETA_G}" \
        --gg_ns_pre_steps "${GG_NS_PRE_STEPS}" \
        --da_beta_min "${DA_BETA_MIN:-0.50}" \
        --da_beta_max "${DA_BETA_MAX:-0.99}" \
        --da_reduction_chunk_elements "${DA_REDUCTION_CHUNK_ELEMENTS:-1048576}" \
        --da_momentum "${DA_MOMENTUM:-true}" \
        --da_use_gg_ns "${DA_USE_GG_NS:-true}" \
        "$@"

    echo "Finished task: ${TASK}"
    echo "=========================================="
done
