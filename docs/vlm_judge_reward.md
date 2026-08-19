# Qwen3.6 vLLM Judge Reward

Wan-Trainer can score DanceGRPO rollouts with `grpo_reward_fn: vbvr_vlm`.
Training ranks VAE-decode the complete generated rollout, encode it as an
in-memory H.264 MP4, and submit an OpenAI-compatible multimodal request to a
standalone vLLM service. vLLM owns the configured temporal sampling pass; the
service owns model weights and KV cache. The reward returns a normalized
scalar in `[0, 1]` and preserves the
existing asynchronous reward submission/resolve ordering.

## Pinned Runtime and Model

The local model snapshot is:

- repository: `Qwen/Qwen3.6-27B`;
- revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`;
- destination: `storage/models/Qwen3.6-27B`;
- verified size: 55,586,107,940 bytes across 29 files;
- runtime: isolated `storage/host_vllm/.venv`, with vLLM 0.26.0, Ray
  2.56.1, PyTorch 2.11.0+cu130, Transformers 5.14.1, and NumPy 2.3.5.

The model and runtime are ignored local artifacts. Recreate them with:

```fish
fish scripts/dev/setup_host_vllm.fish
fish scripts/download/qwen36_27b_hf_mirror.fish
```

The setup helper uses a node-local uv cache and the Tsinghua PyPI mirror by
default because unpacking the CUDA runtime directly under shared QuarkFS is
metadata-bound. It writes the resolved environment to the ignored
`storage/host_vllm/requirements.freeze.txt`. The download helper pins the model
commit, disables the credentialed workstation proxy, uses
`https://hf-mirror.com` with 16 workers, disables Xet for the mirror path, and
finishes with a full remote-file/LFS checksum verification.
Override `WAN_TRAINER_HF_DOWNLOAD_WORKERS` when the shared filesystem or mirror
should receive less concurrency.

## Standalone Single-Node Service

The default server uses tensor parallelism across eight visible GPUs, BF16,
32,768 judge context, at most 32 concurrent sequences, and up to two images
plus one video per request:

```fish
fish scripts/serve/qwen36_27b_vllm.fish
```

In another shell, verify model discovery, vision input, Qwen non-thinking mode,
generic JSON-schema output, and the exact task-prompt/regex contract:

```bash
.venv/bin/python -m src.cli.probe_vlm_service \
  --multimodal-smoke --task-prompt-smoke
```

Important service controls are environment variables:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `WAN_TRAINER_VLM_MODEL_PATH` | `storage/models/Qwen3.6-27B` | Local snapshot |
| `WAN_TRAINER_VLM_PORT` | `18080` | Node-local API port |
| `WAN_TRAINER_VLM_DISTRIBUTED_PORT` | `29501` | Node-local vLLM TP rendezvous |
| `WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE` | `8` | vLLM TP width |
| `WAN_TRAINER_VLM_DATA_PARALLEL_SIZE` | `1` | Total vLLM replica count |
| `WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL` | global DP size | Replicas placed on the service node |
| `WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND` | `mp` | Replica launcher: `mp` or `ray` |
| `WAN_TRAINER_VLM_DISTRIBUTED_EXECUTOR_BACKEND` | DP backend | TP worker launcher: `mp` or `ray` |
| `WAN_TRAINER_VLM_API_SERVER_COUNT` | `1` | API front-end process count |
| `WAN_TRAINER_VLM_GPU_MEMORY_UTILIZATION` | `0.50` | vLLM engine memory budget |
| `WAN_TRAINER_VLM_MAX_MODEL_LEN` | `32768` | Judge context limit |
| `WAN_TRAINER_VLM_MAX_NUM_SEQS` | `32` | Per-DP-replica continuous-batching limit |
| `WAN_TRAINER_VLM_MAX_IMAGES_PER_PROMPT` | `2` | Task mode uses one input image; custom mode can also include the GT final image |
| `WAN_TRAINER_VLM_MAX_VIDEOS_PER_PROMPT` | `1` | One generated MP4 per request |
| `WAN_TRAINER_VLM_RENDERER_NUM_WORKERS` | `1` | Keeps vLLM's multimodal processor cache enabled |
| `WAN_TRAINER_VLM_ENFORCE_EAGER` | `1` | Avoid CUDA-graph memory in the co-hosted smoke |
| `WAN_TRAINER_VLM_GDN_PREFILL_BACKEND` | `triton` | Avoid FlashInfer GDN JIT on the workstation's CUDA 11.1 toolkit |
| `WAN_TRAINER_VLM_USE_FLASHINFER_SAMPLER` | `0` | Use vLLM's native sampler instead of FlashInfer sampling JIT |
| `WAN_TRAINER_VLM_TRITON_CACHE_DIR` | `/tmp/wan-trainer-vllm-triton-cache` | Node-local Triton cache |
| `WAN_TRAINER_VLM_INDUCTOR_CACHE_DIR` | `/tmp/wan-trainer-vllm-inductor-cache` | Node-local Inductor cache |

The API binds to loopback by default. Keep an API exposed across hosts on a
trusted private network and configure authentication; vLLM's distributed
control plane is not intended for an untrusted network.

The host CUDA compiler is 11.1, while the isolated vLLM wheel carries a newer
CUDA runtime. FlashInfer 0.6.14's fallback source build passes an unsupported
`nvcc --threads=1` option on this host, so the launcher defaults to Triton GDN
prefill and the native vLLM sampler. It also keeps all compilation caches under
node-local `/tmp`: eight simultaneous Triton warmups against `~/.triton` on
QuarkFS otherwise spend minutes blocked in shared-filesystem rename/remove
operations. The first local warmup generated about 62 MiB of cache; later
starts reused it.

## Reward Contract

The runnable bounded config is
[`train_dancegrpo_vbvr_pro_5b_384x384x81_vlm_qwen36_smoke_1node_3step.yaml`](../configs/train_dancegrpo_vbvr_pro_5b_384x384x81_vlm_qwen36_smoke_1node_3step.yaml).
The core reward settings are below. This strict smoke deliberately sets
`vlm_reward_fail_on_error: true`; the production multi-node configs use the
fail-open zero fallback documented below.

```yaml
grpo_reward_fn: vbvr_vlm
vlm_reward_base_url: http://127.0.0.1:18080/v1
vlm_reward_model: qwen3.6-27b
vlm_reward_prompt_mode: task_specific
vlm_reward_video_fps: 16
vlm_reward_video_num_frames: 32
vlm_reward_include_gt_first_frame: true
vlm_reward_decode_batch_size: 1
vlm_reward_concurrency: 1
vlm_reward_max_pending_jobs: 1
vlm_reward_max_new_tokens: 1024
vlm_reward_image_max_edge: 512
vlm_reward_use_structured_output: false
vlm_reward_fail_on_error: true
```

The default `task_specific` mode vendors the exact 100-entry `EVAL_PROMPTS`
mapping in
[`vbvr_vlm_eval_prompts.py`](../src/trainer/rewards/vbvr_vlm_eval_prompts.py).
Its SHA-256 is
`4d3159232590bd4b99266c9e82df445a3a54ada50a7af30051cf505057574202`.
The sample task name must match a mapping key; this is intentionally fail-closed
so an unknown task cannot silently receive a generic rubric.

Each task-specific request sends the exact rubric, the input first frame, and
one MP4 containing the complete decoded rollout in chronological order. It
does **not** send the GT final frame: the supplied evaluator prompts define
their input as the first frame and generated video. `vlm_reward_video_fps`
controls the MP4 metadata and is 16 for the 81-frame runs. The training client
does no temporal sampling. Each request instructs vLLM's CPU OpenCV media
loader to select exactly `vlm_reward_video_num_frames: 32` frames uniformly
from the complete MP4, then passes `do_sample_frames: false` to the Qwen HF
processor so it cannot perform a second 2-FPS sampling pass. For an 81-frame
rollout the selected source indices span the complete interval from frame 0 to
frame 80. Qwen's temporal patch size is 2, so those 32 retained frames become
16 visual time groups; temporal patching is model encoding, not another frame
selection pass.

The VLM path does not use the rule scorer's separate 1024x1024 preparation.
`vlm_reward_image_max_edge` is a downscale-only safety bound for both the input
image and MP4 frames: 384/512 inputs at or below it retain native dimensions,
while only oversized inputs are reduced. The input frame is JPEG-encoded and
the rollout is H.264-encoded entirely in memory; both are sent as data URLs, so
no dataset path is exposed to the server. Qwen thinking is disabled per
request.

Each rubric defines different aspect names and weights. The reward derives a
vLLM regex constraint from that task's required output lines, requires every
aspect score, `total_score`, and a nonempty one-line `reason` exactly once,
checks scores are finite and in `[0, 100]`, and verifies that rubric weights sum
to 100. It then normalizes `total_score` to `[0, 1]`. The regex constraint is
automatic in task mode; `vlm_reward_use_structured_output: false` only disables
the fixed generic JSON schema. Regex cannot enforce arithmetic across separate
weight lines, so the final reminder explicitly requires an exact 100-point sum.
If semantic validation still fails, the reward sends the rejected answer and
validation error back to the judge for correction, up to
`vlm_reward_max_retries`. In production, an exhausted retry budget returns
`vlm_reward_error_score` (normally zero) for only that rollout and lets the
distributed step continue.

Set `vlm_reward_prompt_mode: custom` and optionally
`vlm_reward_system_prompt_path` to use the compatibility protocol instead. That
mode interleaves task text, first frame, GT final frame, and generated video,
and can use the fixed score/reason JSON schema. Prompt or rubric changes alter
the optimization target and should be pinned and recorded like scorer changes.

`vlm_reward_fail_on_error: false` with `vlm_reward_error_score: 0` is the
production default: malformed output, exhausted request retries, missing
references, non-finite values, and scores outside the rubric become zero for
the affected rollout. Failures are warning-logged on the scoring rank. This can
flatten or bias group advantages when failures are frequent, so monitor those
warnings. Strict smoke tests may set `vlm_reward_fail_on_error: true` to stop on
the first judge-contract failure.

## Offline Judge Evaluation Of Generated Videos

Use
[`evaluate_vlm_judge_multinode.fish`](../scripts/eval/vbvr_pro/dancegrpo_vlm_qwen36_512x512x81/evaluate_vlm_judge_multinode.fish)
to apply the training-time task-specific Qwen contract to formal VBVR-Pro
videos that have already been generated. This is a judge-only stage: it never
loads Wan, converts a checkpoint, runs diffusion inference, or modifies the
formal EvalKit result tree.

For the native-512 Qwen3.6 DanceGRPO run, the preferred incremental entry point
is now a single command:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_vlm_qwen36_512x512x81/evaluate_incremental_multinode.fish \
  formal --nproc 8
```

After strict formal evaluation completes across all scheduler nodes, that
adapter automatically invokes this offline judge on the same frozen checkpoint
snapshot. Exact cell filters keep every node's shard stable; completed cells
are skipped, partial cells resume, and a node with no pending judgments does
not load Qwen. Pass `--no-vlm-judge` only when an EvalKit-only run is intended.
When invoked outside a scheduler on one machine, omit both `WORLD_SIZE` and
`RANK`; the adapter defaults them to `1` and `0`, respectively. `--nproc` still
selects the local GPU count. Multi-machine invocations must set both variables.
The standalone launcher below remains available for arbitrary compatible
formal roots.

For one machine, omit scheduler variables:

```fish
fish scripts/eval/vbvr_pro/dancegrpo_vlm_qwen36_512x512x81/evaluate_vlm_judge_multinode.fish \
  score \
  --input-root storage/eval_out/vbvr_pro_main_v2_512x512x81_manifest_rl_fujian_new_e140_lr5e6_eval500_181e2010_manifest_afab352e_evalkit_4cc7d028 \
  --concurrency 16
```

For multiple machines, run the same command on every node with `WORLD_SIZE`
set to the evaluation machine count and `RANK=0..WORLD_SIZE-1`. Cells are
round-robin sharded by a deterministic sorted name list; the evaluation
topology is independent of the source training topology. Rank 0 waits for all
selected cells and writes the aggregate. Use `--assignment-only` to perform a
strict, read-only source/resume audit without starting Qwen. Concurrent writers
are valid only when scheduler assignment or explicit `--cell` filters make
their cell sets disjoint; never let two clients own the same cell.

The wrapper starts the same node-local DP4 x TP2 Qwen service and 50% memory
budget as the production training launcher. It uses a 1,800-second startup
timeout because four replicas cold-reading the 51.75-GiB snapshot from QuarkFS
can exceed the co-hosted training timeout. Set `VLM_JUDGE_START_SERVICE=0` to
reuse an already-managed endpoint through `WAN_TRAINER_VLM_BASE_URL`.

The offline client shares its message and request-payload builders with
`VBVRVLMReward`: pinned per-task rubric, input first frame only, no GT final
frame, native 512 maximum edge, non-thinking Qwen, temperature 0/seed 0,
per-task regex output, semantic weight validation/repair, and uniform 32-frame
vLLM sampling. It base64-embeds the existing native MP4 byte-for-byte instead
of decoding and lossy re-encoding it. The request label retains the known
81-frame/16-FPS source contract.

Outputs live in a separate suffix root by default. Every cell contains
`metadata.json`, append-only `samples.jsonl`, `summary.json`, and an
EvalKit-layout `final_scores.txt`. Successful samples survive interruption;
error records are retried on the next invocation. A cell is complete only when
all expected samples have non-error responses under matching source and judge
fingerprints. Rank 0 writes global `summary.json`, `summary.csv`, and
`summary.md`. Running either `score` or `summarize` also backfills a missing
per-cell `final_scores.txt` without contacting Qwen when the judgments are
already complete. The recorded contract includes
the prompt, protocol/evaluator source hashes, Qwen revision, vLLM version,
media settings, and source generation/eval fingerprints; the API key is never
persisted.

### Measured full native-512 matrix

On 2026-08-13, one eight-H800 node judged the complete rule-trained native-512
sampler matrix: baseline plus checkpoints 100 through 2300, six samplers, and
500 videos per cell. All 144 cells and 72,000 unique samples completed under
judge contract `5eeb2e1f3bc2e677daad6858dfab0a8f333741ac393fbb9b6eb4e007540f096a`.
There were zero HTTP retries, semantic repair retries, error fallbacks, missing
responses, or duplicate JSONL records. Consequently, valid zero scores must
not be counted as service failures. The result tree occupies about 121 MiB.

The mean over all 72,000 judgments was `0.587300`. Averaging all six samplers
per model, checkpoint 2200 was best at `0.601000`, versus `0.559937` for the
baseline (`+0.041063` absolute); checkpoint 2300 was `0.598543`. The best
individual cell was checkpoint-2200 Euler ODE at `0.615940` Overall
(`0.694400` In-Domain / `0.537480` Out-of-Domain), narrowly ahead of
checkpoint-2200 CPS 0.9 at `0.615500`. Across all 24 model states, CPS 0.9 had
the highest sampler mean (`0.598998`), followed by CPS 0.7 (`0.595707`), UniPC
ODE (`0.591114`), Euler ODE (`0.586043`), CPS 0.3 (`0.583364`), and CPS 0.1
(`0.568575`). VLM and EvalKit Overall cell means had Pearson correlation
`0.836728` across the 144 matched cells, so the broad trend agrees while the
best sampler choice differs.

The node-local service used DP4 x TP2 at a 50% memory budget and occupied about
42 GiB per card. One 16-request client sustained about 1.9 judgments/s; six
explicitly disjoint clients (96 total requests, 24 per DP replica) sustained
about 5.5-6.1 judgments/s without retries. Keep the wrapper's conservative
default at 16 unless a judge-only node has comparable capacity, and coordinate
disjoint cell ownership before adding clients.

## Co-hosted 50/50 Operation

Run the following command on every scheduler node with the same `MASTER_ADDR`,
`WORLD_SIZE`, `RANK`, and optional `MASTER_PORT` contract used by
`grpo_multinode.fish`:

```fish
fish scripts/train/grpo_vlm_eval_multinode.fish --nproc 8 \
  --config configs/train_dancegrpo_vbvr_pro_5b_384x384x81_vlm_qwen36_smoke_1node_3step.yaml
```

The generic wrapper starts one DP1 x TP8 Qwen endpoint inside each node/pod,
runs both a real image/JSON probe and an exact task-rubric/video/regex probe,
delegates training to the standard multi-node GRPO launcher, and terminates the
whole vLLM process group on normal exit or signals. Every node uses the same
loopback URL, so no scheduler-specific service discovery is needed. The
cluster launcher overrides this generic topology to DP4 x TP2 on every node.

`--gpu-memory-utilization 0.50` is not a hard CUDA partition. It budgets vLLM's
weights, KV cache, and engine workspace against total device memory; PyTorch
training can still allocate until the card is exhausted. Co-hosting also runs
vLLM TP collectives and training FSDP collectives in separate processes on the
same GPUs. Treat 50/50 as an operational target that requires an end-to-end
smoke at the production resolution, model, replay shape, and request
concurrency. Lower the vLLM fraction or sequence/context limits if combined
peak memory is too close to the card limit.

### Measured single-node result

The measurements below predate the direct-MP4 request path and used one input
image plus six generated-frame JPEGs. They remain topology and co-hosting
evidence, but do not establish latency, memory, or reward equivalence for the
current fixed-32-frame video request. Re-run a monitored smoke before using
the new path for production training.

On 2026-08-05, the bounded config completed three real optimizer steps on one
node with eight 81,559 MiB H800s. It used the merged 5B full-fine-tuning model,
FSDP8, 384x384x81, one shared prompt, G=8, T=8, Flow-CPS 0.7, VAE decoding,
and one Qwen request per rank. The launcher exited zero and cleaned up every
training and vLLM process.

- Idle TP8 vLLM used 33,017 MiB per card (40.48%), including a 23.44 GiB KV
  cache per card.
- The combined `nvidia-smi` peak was 59,477 MiB per card (72.92%), leaving
  22,082 MiB (27.08%) physical headroom. The training process itself reported
  a 19.7/23.1 GiB PyTorch allocated/reserved peak; that counter excludes the
  separate vLLM process.
- Step times were 20.79, 22.56, and 21.52 seconds. Step rewards were
  `1.0000+/-0.0000`, `0.0000+/-0.0000`, and `0.1250+/-0.3536`; step 3 had a
  nonzero gradient norm of 0.0022, proving that the judge produced a usable
  group-relative update rather than only a zero-advantage plumbing pass.
- The three batches exercised the exact G-21, O-52, and G-45 rubrics. All
  responses matched their task-dependent field sets and parsed successfully.
- All 24 audit rollouts decode successfully as 384x384, 81-frame, 16 FPS,
  5.0625-second videos beneath
  `storage/host_vllm/runs/dancegrpo_vbvr_pro_5b_384x384x81_vlm_qwen36_task_prompts_strict_smoke_1node_3step/rollout_videos`.

An earlier unconstrained task-prompt attempt correctly failed closed on G-45:
some ranks emitted a long analysis and exhausted the output budget before the
required fields. The per-task regex, final output reminder, and 1024-token cap
were added before the successful run above. Do not remove the constraint merely
because shorter rubrics often happen to answer in the requested format.

This establishes feasibility for the bounded 5B shape, not for G=32/T=30,
512x512 production training, A14B, higher judge concurrency, or a multi-node
optimizer step. Repeat the monitored smoke at the intended production shape;
the remaining memory margin is capacity evidence, not a fixed 60% reservation.

### Measured local DP scale-out

The judge workload is heterogeneous: the 100 rubrics have different output
fields and generated lengths. A single TP8 engine therefore has head-of-line
tail even when every training rank submits the same number of requests. On the
same eight-H800 node and 40% vLLM memory budget, a mixed benchmark used 32
exact task-rubric requests, one input image plus six generated 384-pixel
frame JPEGs, a 1024-token cap, and client concurrency 16:

| vLLM topology | Backend | Idle MiB/GPU | Throughput | Mean | p95 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DP1 x TP8 | `mp` | 33,019 | 2.151 req/s | 5.903 s | 9.428 s | 10.451 s |
| DP2 x TP4 | `mp` | 32,863 | 2.012 req/s | 6.331 s | 10.159 s | 11.103 s |
| DP4 x TP2 | `mp` | 32,643 | 2.497 req/s | 4.723 s | 7.144 s | 8.127 s |

DP2 did not help, but DP4 improved throughput by 16.1% and reduced p95 by
24.2% relative to TP8. At concurrency 32 with 64 requests, DP4 reached 5.184
req/s versus TP8's 4.015 req/s, a 29.1% improvement. `max_num_seqs` is applied
per replica, but the TP2 engine's measured full-32K-context KV capacity was
only 4.09 sequences per replica; the actual judge inputs are much shorter and
the configured workload is 16 concurrent requests per node.

The same DP4 x TP2 shape also passed the strict vision/task probes using Ray
for both DP and TP actor placement. One benchmark trial produced 2.633 req/s
and 6.321-second p95, within the same performance range as multiprocessing;
cache warmth and startup order make that small difference non-actionable. Ray
startup was materially slower and required explicit cluster cleanup, so it is
not the default merely because its one warm run was faster.

A real co-hosted DP4 x TP2 multiprocessing smoke then completed one FSDP8 5B
optimizer step at 384x384x81, G=8, T=8 and exited cleanly. The combined
one-second-sampled physical peak was 50,953 MiB/GPU, and all eight rollout
videos decoded correctly. The G-21 judge returned the same score for every
group member, so this run had zero group advantage and is memory/lifecycle
evidence only; the earlier three-step TP8 run remains the nonzero-update proof.
The shapes differ, so the lower peak than the TP8 three-step run must not be
interpreted as an exact topology-only memory saving.

### Native-512 and 50% memory smoke

On 2026-08-06, DP4 x TP2 at `gpu_memory_utilization=0.50` passed both strict
service probes and a real co-hosted one-step run. Idle vLLM used
40,770-40,774 MiB/card. Each TP2 replica reported 25.69 GiB weights, 12.34 GiB
KV cache, and 11.44 full-32K-context sequences of theoretical capacity.

The training side used FSDP8 full-FT Wan 5B, 512x512x81, G=8, T=8 and sent the
input plus six generated frames at native 512 resolution. The optimizer step
finished in 36.39 seconds and the complete process exited zero. Training
reported 15.5/15.8 GiB allocated/reserved; periodic physical samples reached
51,483 MiB/card, but were not a one-second peak trace. All eight saved videos
decoded as 512x512, 81 frames, 16 FPS. The single G-21 group received identical
1.0 rewards, so this proves native-resolution request, memory, and lifecycle
feasibility rather than a nonzero update. Production G32/T30 still requires a
monitored one-step launch.

### 4/8/16-node candidate

The production-style candidate is
[`train_dancegrpo_vbvr_pro_5b_512x512x81_vlm_qwen36_cps_from_nsft_bs_32_lr_5e-6_manifest_rl_multinode.yaml`](../configs/train_dancegrpo_vbvr_pro_5b_512x512x81_vlm_qwen36_cps_from_nsft_bs_32_lr_5e-6_manifest_rl_multinode.yaml).
The lower-pressure native-384 counterpart is
[`train_dancegrpo_vbvr_pro_5b_384x384x81_vlm_qwen36_cps_from_nsft_bs_32_lr_5e-6_manifest_rl_multinode.yaml`](../configs/train_dancegrpo_vbvr_pro_5b_384x384x81_vlm_qwen36_cps_from_nsft_bs_32_lr_5e-6_manifest_rl_multinode.yaml);
it uses rollout/replay/VAE chunks of eight and never enlarges judge frames.
Run the same command on every node with scheduler-provided `WORLD_SIZE=4`, `8`,
or `16`, `RANK=0..WORLD_SIZE-1`, and a shared `MASTER_ADDR`:

```fish
fish scripts/train/grpo_vlm_eval_cluster.fish \
  --yaml=configs/train_dancegrpo_vbvr_pro_5b_512x512x81_vlm_qwen36_cps_from_nsft_bs_32_lr_5e-6_manifest_rl_multinode.yaml
```

The cluster wrapper starts with the Fujian project path, requires an explicit
`--yaml=<path>`, detects the three supported node counts, pins the shared FA3
cache, and translates the YAML to the trainer's internal `--config` argument.
It starts four TP2 replicas behind one node-local vLLM DP endpoint using a 50%
per-card budget. It automatically suffixes checkpoint and W&B names with
`nodes4_world32`, `nodes8_world64`, or `nodes16_world128`; explicit CLI
overrides still win. `WAN_TRAINER_VLM_LAUNCH_DRY_RUN=1` resolves and prints the
topology without starting a service. The old `grpo_vlm_eval_4node.fish` path is
only a compatibility delegate to this launcher.

It retains the Fujian optimizer, bs32/G32/T30 Flow-CPS schedule, Liger, pinned
FA3, Inductor, delayed replay, and full fine-tuning. At world32/64/128 it uses
4/8/16 HSDP replicas x eight shards and two 16-prompt waves. Ranks per prompt
are 2/4/8, so each rank owns 16/8/4 rollouts per wave. The candidate caps
rollout/replay/VAE chunks at four, offloads T5/VAE, and judges native 512
frames rather than preparing 1024x1024 inputs.

Before allowing a long auto-resuming job, use a fresh output override and
`--max_steps 1 --save_steps 0` to measure the real combined peak. Delayed replay
may overlap vLLM TP with training HSDP collectives; if the first trace shows
collective contention, compare a run with `--no-grpo_delayed_replay` rather
than changing the reward or optimizer at the same time.

## Multi-Node DP and Tail Latency

The first multi-node run should use the validated local design: four TP2 Qwen
replicas per node, vLLM multiprocessing DP inside the node, and one loopback
endpoint per node. This keeps TP collectives on NVLink, removes most of the
single-engine queue tail, and avoids sending base64 image payloads across the
network.

There can still be a cross-node straggler. World32/64/128 assigns 256/128/64
judge requests to each node per optimizer step, but different task rubrics and
generated outputs take different time. A node-local vLLM coordinator can
move work among its four replicas, but it cannot give that node's remaining
requests to a replica on another node; the distributed training step waits for
the slowest rank/node. Measure per-node `reward_drain` and service queue/latency
before adding cluster-wide scheduling.

vLLM's internal DP exposes one endpoint and supports dense models combined
with TP. In vLLM 0.26 its dispatcher scores each engine from running and
waiting queue lengths; it is not yet KV-cache-aware. `--max-num-seqs` applies
to every DP rank, and a large DP deployment can scale its front end with
`--api-server-count` while keeping one HTTP port.

### Cluster-wide Ray option

Ray can place a global DP(4 x nodes) x TP2 deployment across the same cluster.
Ray is responsible for actor/placement-group scheduling; vLLM's API
front end still chooses the replica for each request. With the vLLM 0.26
default `VLLM_RAY_DP_PACK_STRATEGY=strict`, each TP2 placement group stays on
one node, and `data_parallel_size_local=4` requires four such replicas on the
head node and each selected peer. Start the Ray cluster from the isolated vLLM
environment, then run one service command on the head:

```fish
# Head node; use the scheduler-visible private IP.
storage/host_vllm/.venv/bin/ray start --head --node-ip-address=$HEAD_IP \
  --port=6379 --num-gpus=8

# Each worker node.
storage/host_vllm/.venv/bin/ray start --address=$HEAD_IP:6379 \
  --node-ip-address=$THIS_NODE_IP --num-gpus=8

# Head node only: one endpoint, four globally scheduled replicas per node.
set -lx WAN_TRAINER_VLM_DATA_PARALLEL_SIZE (math "$WORLD_SIZE * 4")
env RAY_ADDRESS=auto VLLM_RAY_DP_PACK_STRATEGY=strict \
  WAN_TRAINER_VLM_HOST=0.0.0.0 \
  WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE=2 \
  WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL=4 \
  WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND=ray \
  WAN_TRAINER_VLM_DISTRIBUTED_EXECUTOR_BACKEND=ray \
  WAN_TRAINER_VLM_API_SERVER_COUNT=4 \
  fish scripts/serve/qwen36_27b_vllm.fish
```

For this mode, launch training with
`WAN_TRAINER_VLM_START_SERVICE=0` everywhere and set
`WAN_TRAINER_VLM_BASE_URL=http://$HEAD_IP:18080/v1` on every node. Ray sees the
vLLM actors as owning all GPU resources, while the co-hosted `torchrun`
processes are outside Ray; the 50%/remaining-memory agreement is therefore
still operational rather than enforced. Restrict this endpoint and Ray ports
to the trusted cluster network. The training wrapper intentionally rejects
wrapper-managed Ray because Ray actors do not share the vLLM launcher's process
group. Stop the service first, then run
`storage/host_vllm/.venv/bin/ray stop --force` on every node and verify that no
Ray or vLLM process remains before releasing the allocation.

This global queue lets an idle replica on one machine take requests that would
otherwise remain behind a slow local queue, but it also centralizes API work,
sends images over the network, adds Ray startup/cleanup state, and can overlap
vLLM and HSDP communication across nodes. The local Ray DP4 x TP2 smoke passed;
global Ray plus co-hosted multi-node training has not. Use it only as a
measured follow-up if local DP shows meaningful between-node tail.

Ray is not required to get a global vLLM queue. Native multiprocessing internal
DP can launch the head with global DP(4 x nodes)/local DP4 and launch the other
nodes in `--headless` mode with start ranks increasing by four plus a shared DP
address/RPC port. It offers the same single-endpoint routing with less
cluster-runtime state, at the cost of coordinated service commands. The official data
parallel deployment guide contains the exact two-node form to extend.

## References

- [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B)
- [vLLM 0.26.0 serve CLI](https://docs.vllm.ai/en/v0.26.0/cli/serve/)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/v0.26.0/serving/parallelism_scaling/)
- [vLLM data-parallel deployment](https://docs.vllm.ai/en/v0.26.0/serving/data_parallel_deployment/)
- [Ray cluster startup](https://docs.ray.io/en/latest/cluster/getting-started.html)
