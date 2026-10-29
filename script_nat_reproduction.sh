#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
PYTHON=${PYTHON:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-11234}
SEED=${SEED:-0}
PT_BATCH_SIZE=${PT_BATCH_SIZE:-32}
ACCUM_ITER=${ACCUM_ITER:-1}
LP_BATCH_SIZE=${LP_BATCH_SIZE:-128}
PT_EPOCHS=${PT_EPOCHS:-400}
LP_EPOCHS=${LP_EPOCHS:-100}
OUTPUT_ROOT=${OUTPUT_ROOT:-./output_dir/nat_ntu60_xsub_seed${SEED}}
STAGE=${STAGE:-both}
case "$STAGE" in both|pt|lp) ;; *) echo 'STAGE must be both, pt, or lp' >&2; exit 2 ;; esac

# The original motion baseline is already reproduced; do not rerun it by default.
if [[ $# -eq 0 ]]; then set -- joint_patch joint_nat; fi
for experiment in "$@"; do
    case "$experiment" in
        motion_patch|joint_patch|joint_nat|joint_nat_cl) ;;
        *) echo "Unknown experiment: $experiment" >&2; exit 2 ;;
    esac
    config="./config/ntu60_xsub_joint/nat/pretrain_${experiment}.yaml"
    pt="$OUTPUT_ROOT/$experiment/pt"
    lp="$OUTPUT_ROOT/$experiment/lp"
    checkpoint="$pt/checkpoint-$((PT_EPOCHS - 1)).pth"
    if [[ "$STAGE" != lp && -e "$pt/log.txt" ]] || [[ "$STAGE" != pt && -e "$lp/log.txt" ]]; then
        echo "Existing logs for $experiment. Choose a new OUTPUT_ROOT, or use STAGE=lp after PT." >&2
        exit 2
    fi
    if [[ "$STAGE" != lp ]]; then
        "$PYTHON" -m torch.distributed.launch --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" main_pretrain.py --config "$config" --batch_size "$PT_BATCH_SIZE" --accum_iter "$ACCUM_ITER" --epochs "$PT_EPOCHS" --seed "$SEED" --output_dir "$pt" --log_dir "$pt/tensorboard"
    fi
    if [[ "$STAGE" != pt ]]; then
        if [[ ! -f "$checkpoint" ]]; then echo "Missing PT checkpoint: $checkpoint" >&2; exit 2; fi
        "$PYTHON" -m torch.distributed.launch --nproc_per_node="$NPROC_PER_NODE" --master_port="$((MASTER_PORT + 1))" main_linprobe.py --config ./config/ntu60_xsub_joint/linprobe_t120_layer8.yaml --finetune "$checkpoint" --batch_size "$LP_BATCH_SIZE" --accum_iter 1 --epochs "$LP_EPOCHS" --lr 0.1 --seed "$SEED" --output_dir "$lp" --log_dir "$lp/tensorboard" --dist_eval
    fi
done
"$PYTHON" tools/summarize_nat_runs.py --root "$OUTPUT_ROOT"
