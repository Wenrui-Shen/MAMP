# MAMP -> NAT -> NAT + CL: NTU-60 X-sub

## Baseline and experimental order

The supplied original MAMP logs contain 400 PT epochs and 100 LP epochs.
Best LP top-1 is **84.940562850234% at epoch 95**; last-epoch LP top-1 is
**84.78893742568282%**. PT ends with loss 0.2434731974793128.
`experiments/nat/baseline_summary.json` stores these values, source paths, and
SHA-256 hashes. The protocol comes from the conversation; the JSON logs alone
do not establish every launch argument or the originating checkpoint.

Run the normalization controls first, then CL. Every new experiment trains
from scratch; NAT + CL does not add another 400 epochs onto a trained NAT model.

| Run name | Reconstruction | Target normalization | CL | Paper NTU-60 X-sub LP reference |
| --- | --- | --- | --- | --- |
| `motion_patch` | Motion | Original patch normalization | No | 84.9% |
| `joint_patch` | Joint coordinates | Original patch normalization | No | 74.8% |
| `joint_nat` | Joint coordinates | NAT | No | 85.1% |
| `joint_nat_cl` | Joint coordinates | NAT | Yes | 86.9% |

Paper references are Tables 5 and 6 of Gong et al.,
*Rethinking Masked Data Reconstruction Pretraining for Strong 3D Action
Representation Learning*, AAAI 2025, pp. 3149-3157:
[official paper](https://ojs.aaai.org/index.php/AAAI/article/download/32324/34479).
The table contains the paper's numbers, not measurements of this implementation.
The already supplied baseline normally does not need to be run again.

## What is held fixed

All four configurations preserve the validated MAMP NTU-60 X-sub protocol:
8 encoder + 3 decoder layers, encoder/decoder width 256, 8 heads, 120 frames,
4-frame temporal patches, 25 joints, 90% masking, motion-aware tau 0.8,
both people including padding, crop [0.5, 1], random rotation enabled,
400 PT epochs, warmup 20, AdamW, LR 0.001 -> 0.0005, weight decay 0.05,
seed 0, two GPUs, 32 videos per GPU, accumulation 1, autocast disabled.

LP uses the existing native `linprobe_t120_layer8.yaml`: frozen online encoder,
native 256 -> 60 linear head, 100 epochs, 128 videos per GPU (256 total),
SGD LR 0.1 -> 0, evaluation crop [0.95], seed 0, `--dist_eval`.
No empty-person filtering, input normalization, new augmentation, projector,
alternative LP head, or modified LR schedule is introduced.

`joint_patch` intentionally uses the same MAMP masking as the motion control.
Using the existing MAE model would also change masking to random, confounding
the target comparison. The new motion/patch path is regression-tested against
native MAMP, and the joint/patch path against native MAE with random masking.

## NAT implementation

Original target patches have shape `[N*M, 750, 12]`. Original normalization
mixes 4 timestamps and 3 coordinates in the last dimension. For NAT, reshape
each token into `[4, 1, 3]`, and calculate mean/variance **only over the
4 timestamps**, independently for every joint and x/y/z coordinate (paper Eq. 5).
Flatten back to 12 values and retain MAMP's masked-token MSE reduction:
average over the 12 prediction coordinates, then average over masked tokens.
The paper's vector-valued Eq. 7 does not spell out coordinate-wise reduction;
this choice preserves the validated MAMP reconstruction loss scale.

This changes only the reconstruction target. Feeder `normalization` remains
False, and the input to the embedding/encoder is unchanged. For example, large
coordinate-dependent offsets disappear under NAT, whereas they can dominate
the original coordinate-mixed target normalization.

The paper does not specify the numerical epsilon or variance estimator.
This implementation inherits MAMP's sample variance (`unbiased=True`) and
epsilon `1e-6`; all-zero/constant targets stay finite and zero. No normalization
is performed over all 120 frames or over a whole dataset. The target modes also
support motion + NAT for a later 2-by-2 ablation, but this is not part of the
paper's Table 5 or the default experiment list.

## CL implementation and choices

Paper-specified parts:

- The online encoder sees only unmasked tokens; the EMA encoder sees all 750
  tokens of the same already cropped/rotated input (no second random crop).
- Both branches use GAP and cosine similarity, without an added projection MLP.
- InfoNCE temperature 0.07; EMA momentum 0.999; FIFO queue capacity 65536;
  total loss `reconstruction + 0.0001 * contrastive` (Eqs. 8-10).
- The teacher is updated by EMA, rather than the reconstruction/CL optimizer.
  Downstream evaluation keeps the online embedding and encoder.

Implementation choices where the paper is not explicit:

- Initialize the teacher by copying the online encoder, freeze its parameters,
  keep it in eval mode, and evaluate it under `no_grad`. EMA updates are specified
  by the paper; exact initialization and these execution-mode details are not.
- One contrastive instance per video: average its two people's pooled features,
  matching native MAMP LP. Do not treat empty-person padding as independent
  contrastive instances. L2 normalization is applied after pooling, with epsilon
  `1e-6`. The paper does not describe its multi-person pooling or cosine epsilon.
- Start the queue empty and enqueue actual teacher features; the first optimizer
  window has zero CL loss. No random initial negatives or sample-ID exclusion
  are added. Previous samples remain negatives as in a plain FIFO queue.
- Update the teacher and enqueue gathered keys only after a successful optimizer
  step. With accumulation, commit once per complete window. Scaler-skipped steps
  and discarded trailing windows do not advance teacher/queue state. All ranks
  gather keys in rank order, giving identical queues.
- `teacher_batch_size: 8` chunks the no-grad teacher over person sequences before
  the online forward. Eval-mode LayerNorm makes chunked features equivalent to
  a full teacher batch within floating-point tolerance, while reducing peak
  memory. This does not change the online batch, teacher mask, or queue size.
- Resume validates saved model options, data options, masking, seed, batch,
  accumulation, world size and schedule. An ablation change needs a new run.
  Full teacher/queue state is saved in the regular checkpoint. As in native
  MAMP, RNG states are not checkpointed, so resume is not bitwise reproducible.

The paper says that other hyperparameters follow MAMP, without publishing all
details in its main text. In particular, it does not settle the official MAMP
3-vs-5 decoder discrepancy or explicitly specify its masking tau. We keep the
already reproduced **8+3/tau0.8/two-person** setup to isolate the changes.
Thus these are controlled reproductions of the paper's mechanisms, not a claim
of exact equivalence to unretrieved author training code. PT batch total 64 is
the user's validated setup; the paper's stated LP batch 256 is matched.

## Run on the training server

Use the project's existing Python 3.8 / torch 1.8.1 / timm 0.3.2 environment.
Launch from a fresh copy of these source changes with data at
`../data/MAMP/ntu/NTU60_XSub.npz` relative to the repository root.

First run joint reconstruction with original normalization, then with NAT;
each finishes PT and LP before the next experiment starts:

```bash
bash script_nat_reproduction.sh joint_patch joint_nat
```

After checking that NAT closes the gap to motion reconstruction, run CL:

```bash
bash script_nat_reproduction.sh joint_nat_cl
```

Each group gets its own PT and LP subdirectory under
`output_dir/nat_ntu60_xsub_seed0/`. GPUs default to 0,1. The script exits on a
failed stage and refuses to append to existing PT/LP logs. To finish LP after
PT-only or after a PT-success/LP-failure run that has not yet written LP logs:

```bash
STAGE=lp bash script_nat_reproduction.sh joint_nat
```

Optional control rerun, independent seeds, and report:

```bash
bash script_nat_reproduction.sh motion_patch
SEED=1 bash script_nat_reproduction.sh joint_patch joint_nat joint_nat_cl
python tools/summarize_nat_runs.py --root output_dir/nat_ntu60_xsub_seed0
```

When a root has no motion baseline run, its summary imports the supplied logs
as a reference, clearly labeled `supplied_baseline_logs`. This is not a motion
run at a new seed: the supplied JSON does not record the seed. For a rigorous
multi-seed comparison, run the matching motion control in each seed's root;
that run takes precedence. Use `--baseline` to select another supplied summary.

For a short GPU integration check, use a separate root and small epochs/batch:

```bash
PT_EPOCHS=1 LP_EPOCHS=1 PT_BATCH_SIZE=2 LP_BATCH_SIZE=2 OUTPUT_ROOT=output_dir/nat_gpu_smoke bash script_nat_reproduction.sh joint_nat_cl
```

This still iterates the entire training split for one PT/LP epoch; it is an
integration check, not an accuracy experiment. To reduce memory for a real run,
set `PT_BATCH_SIZE=16 ACCUM_ITER=2` consistently across comparison groups and
use a new output root. For seed comparisons, rerun the matching motion control.

Judge the target ablation by **LP accuracy**, not PT loss magnitudes: changing
the normalization also changes the scale/meaning of reconstruction loss.
The summarizer records both best and last LP accuracy, and PT/LP epoch counts,
so incomplete runs or checkpoint-selection differences stay visible.
`joint_nat_cl` additionally logs reconstruction loss, unweighted CL loss and
queue occupancy; do not mistake the small weight for a zero/disabled CL loss.

## Verification and limits

Run the tests in the existing training environment:

```bash
python -m unittest discover -s tests -p 'test_nat*.py' -v
NAT_RUN_DDP_TESTS=1 python -m unittest discover -s tests -p 'test_nat*.py' -v
```

The second command includes a real two-process CPU Gloo check, including a
second update with a populated negative queue. Local testing
uses a temporary torch 2.5.1 CPU installation and a test-only `torch._six` shim
when executing the unchanged torch-1.8-era training utilities. New production
code retains Python 3.8 and torch 1.8 APIs. All 25 tests passed locally, including
the two-process check; Python 3.8 syntax, Bash syntax, and diff checks also passed.
No new 400-epoch training or GPU accuracy result is claimed.
The local machine lacks the prepared NTU dataset and has a 10 GiB GPU, so
full NTU-60 runs must be performed in the existing training environment.
