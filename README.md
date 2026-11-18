# lerobot-fastload

Train LeRobot policies (pi0.5, GR00T, ACT, diffusion, ...) without decoding video during training.

LeRobot datasets store camera streams as mp4 (AV1/H.264). The stock training loop decodes every
sample on CPU DataLoader workers, with a seek to a keyframe per frame. On fast multi-GPU nodes the
GPUs end up waiting for the CPUs. fastload decodes each video **once** into a uint8 cache and then
feeds training from RAM and VRAM:

```
mp4 ──(once)──► uint8 cache on local disk ──► page cache (RAM, shared by all ranks)
                                         └──► VRAM tier: as many of this rank's episodes as fit,
                                              sized per run from the measured peak memory
```

* **Same batches as LeRobot.** Frames are decoded by LeRobot's own `_query_videos` and stored before
  the `/255` step, so cached frames are bit-identical. Action chunks, `*_is_pad`, observation history,
  task strings and image transforms follow `LeRobotDataset.__getitem__`. `tests/test_equivalence.py`
  checks this against your dataset.
* **No training-code changes.** `fastload.train` runs LeRobot's `lerobot_train.train()` with the data
  loader swapped. Checkpoints and `train_config.json` are plain LeRobot.
* **Same batch order and randomness as LeRobot** (default `order=lerobot`). Batches are drawn by a real
  DataLoader over row indices that accelerate prepares exactly like LeRobot's, so every rank gets the
  same batches in the same order and the process RNG advances identically (this matters for policies
  that sample on the CPU during training, e.g. SmolVLA and GR00T). Image-augmentation parameters are drawn
  from the RNG streams LeRobot's `num_workers` DataLoader workers would have, and applied on the GPU.
  `tests/test_lerobot_order.py` checks batches and RNG states against LeRobot.
* **Adapts to each run.** Nothing to tune per GPU count: each rank measures its memory after a few steps
  and moves as many frames to VRAM as fit. Frames not in VRAM are gathered from the cache by a background
  thread and copied on a side CUDA stream.

Tested with LeRobot 0.4.2 and 0.4.4.

## Install

Into an environment that already has LeRobot (>=0.4.2, <0.5):

```bash
pip install git+https://github.com/eunyoung030927/lerobot-fastload.git
# or, to edit / run the tests and benchmarks:
git clone https://github.com/eunyoung030927/lerobot-fastload.git
cd lerobot-fastload && pip install -e .
```

## 1. Build the cache (once per dataset)

```bash
python -m fastload.build_cache --repo_id <repo_id> --root <dataset dir> \
    --cache_dir /nvme/fastload/<name> --num_workers 32
```

* Size = frames × cameras × C×H×W bytes (e.g. 42k frames × 2 × 480×640×3 ≈ 77 GB; 934k frames of
  256×256 with 2–3 cameras ≈ 466 GB). Put it on local NVMe, not a network mount.
* Resumable: rerun the same command after an interruption. `--verify N` re-checks N random frames
  against LeRobot bit-exactly (default 32).
* The cache is tied to the dataset's `meta/info.json`; a changed dataset needs a rebuild.

## 2. Train

Use the usual `lerobot-train` arguments, change the entry point, and add the cache:

```bash
accelerate launch --num_processes 8 -m fastload.train \
    --dataset.repo_id=<repo_id> --dataset.root=<dataset dir> \
    --policy.type=pi05 --batch_size=32 --steps=30000 \
    --fastload.cache_dir=/nvme/fastload/<name>
```

| option | default | meaning |
|---|---|---|
| `--fastload.cache_dir` | required | cache built in step 1 |
| `--fastload.order` | `lerobot` | `lerobot` = LeRobot's batches, order and data RNG (keep `--num_workers` as in your LeRobot runs); `shard` = each rank owns whole episodes (largest VRAM hit rate on many GPUs, different order) |
| `--fastload.loader` | `fast` | `fast` = FastLoader; `dataloader` = LeRobot's DataLoader reading the cache (no VRAM tier) |
| `--fastload.placement` | `auto` | `auto` grows a VRAM tier after warm-up; `gpu` puts the whole shard in VRAM at start; `host` never uses VRAM |
| `--fastload.warmup_steps` | 20 | steps before measuring memory (`auto`) |
| `--fastload.gpu_margin_gb` | max(2 GB, 5%) | VRAM kept free above the measured peak |
| `--fastload.max_gpu_cache_gb` | none | cap on the VRAM tier |
| `--fastload.prefetch` | 2 | batches prepared ahead |
| `--fastload.memory_guard` | true | every step, drop VRAM-tier chunks if the run's peak memory grew into the margin |
| `--fastload.oom_retry` | true | CUDA OOM in a step: one process -> halve the VRAM tier and retry; several -> stop all ranks at once and leave a note for `fastload.supervise` |
| `--fastload.augment_device` | `gpu` | where image augmentation runs: `gpu` = fast, augmented pixels differ from LeRobot by float rounding (~1e-6); `cpu` = LeRobot's exact CPU path in the prefetch thread, bit-identical, no VRAM tier |
| `--fastload.decode_hook_device` | `cpu` | where a wrapper around LeRobot's `decode_video_frames` (e.g. a data-loader resize) runs: `cpu` = bit-identical to LeRobot, `gpu` = faster, float rounding |

With `loader=fast` no worker processes are started: `num_workers` only selects the emulated worker RNG
streams for image augmentation, and `pin_memory` / `prefetch_factor` are ignored.

## Custom training scripts

```python
import fastload

# after creating the Accelerator (rank / world size / device are read from it);
# same arguments as the DataLoader it replaces, including num_workers
dataloader = fastload.make_loader(dataset, batch_size, cache_dir="/nvme/fastload/<name>",
                                  sampler=sampler, shuffle=sampler is None, num_workers=4)
policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(policy, optimizer, dataloader, lr_scheduler)
for batch in dataloader:   # one epoch of this rank's batches; tensors already on the GPU
    ...
```

`dataset` is your `LeRobotDataset` (delta timestamps and image transforms are taken from it).
`fastload.CachedLeRobotDataset.from_dataset(dataset, cache_dir)` alone gives a drop-in dataset for an
ordinary `DataLoader` (any `num_workers`) if you only want to skip video decoding.

## Benchmark and test on your data

```bash
python tests/test_equivalence.py --repo_id <repo_id> --root <dir> --cache_dir <cache>
python benchmarks/bench_loader.py --repo_id <repo_id> --root <dir> --cache_dir <cache> --batch 32
torchrun --nproc_per_node 2 benchmarks/bench_loader.py ... --modes fast_host,fast_gpu
```

The benchmark measures data loading only. Whether training gets faster depends on whether the GPU was
waiting for data: compare LeRobot's `data_s` with `updt_s` in the training log before switching.

## Batch order and randomness

`order=lerobot` (default) reproduces LeRobot's data pipeline:

| LeRobot | fastload |
|---|---|
| sampler (`RandomSampler` or `EpisodeAwareSampler`), accelerate batch sharding and RNG sync | the same objects, iterated over row indices (`accelerator.prepare` prepares them) |
| `base_seed` and sampler seed drawn from the process RNG when each epoch starts | the same draws, at the same point |
| worker `w` reseeds `torch` / `numpy` / `random` from `base_seed + w`; batch `k` is fetched by worker `k % num_workers` | the same streams, emulated in the main thread (no worker processes) |
| `num_workers=0`: transforms draw from the process RNG, one batch ahead (accelerate prefetches) | the same |
| `image_transforms` per sample and camera on the CPU | same parameters; kernels on the GPU (float rounding, ~1e-6) or, with `--fastload.augment_device=cpu`, LeRobot's CPU path (bit-identical) |

With several GPUs every rank's batches come from the whole dataset, so the VRAM tier is a random subset
of all frames and the rest is read from RAM / disk each batch.

`order=shard` instead gives each rank whole episodes (round-robin after sorting episodes by task),
shuffled within the shard every epoch. Every optimizer step still averages gradients over samples from
all shards, but the batches differ from LeRobot's. Use it when VRAM hit rate on many GPUs matters more
than reproducing LeRobot.

## Running out of GPU memory

The VRAM tier is sized from the peak memory measured during the first `warmup_steps`. Later steps can
need more (longer sequences, evaluation, checkpointing), so:

1. **Memory guard** (every step, any number of GPUs): if free VRAM minus the run's peak transient need drops
   below half of `gpu_margin_gb`, whole tier chunks (1 GB each) are dropped until the margin is back. No
   copies; dropped frames are read from RAM / disk again. Batches are unchanged.
2. **OOM in a step, one process**: halve the tier and retry the step.
3. **OOM in a step, several processes**: the other ranks already wait in the gradient all-reduce, so one rank
   cannot redo the step alone. All ranks stop at once (instead of hanging until the NCCL timeout) and leave
   a note. Run training under the supervisor to restart automatically from the last checkpoint (LeRobot's
   own `--resume`, including RNG state) with half the tier:

```bash
python -m fastload.supervise --max_restarts 3 -- \
    accelerate launch --num_processes 8 -m fastload.train <lerobot-train args> --fastload.cache_dir=...
```

If there is no checkpoint yet, the run starts over with the smaller tier and the failed output directory
is renamed to `<output_dir>.oom<N>`.

## Data-loader code that wraps LeRobot's video decoder

Training scripts sometimes change frames right after decoding by wrapping
`lerobot.datasets.lerobot_dataset.decode_video_frames` (e.g. resizing 480x640 to 240x320 in the DataLoader
workers). fastload never decodes video, so it calls that wrapper itself with LeRobot's backend decoder
answering with the cached frames: the wrapper's own code runs on exactly the frames LeRobot's decoder would
return, at the same point (before `image_transforms`). With `--fastload.decode_hook_device=cpu` (default)
the result is bit-identical to LeRobot; those cameras then skip the VRAM tier. `gpu` runs the wrapper on
the batch on the GPU instead. Build the cache in a process without the wrapper (`build_cache` refuses
otherwise).

## Limitations

* Single `LeRobotDataset` (as `lerobot-train`); `StreamingLeRobotDataset` is not supported.
* Image transforms must be torchvision v2 transforms (LeRobot's are). Their parameters match LeRobot's; with
  the default `augment_device=gpu` augmented pixels can differ from LeRobot's CPU result by float rounding
  (`augment_device=cpu` makes them bit-identical at the cost of the VRAM tier and CPU time).
* `cudnn.benchmark` (on in lerobot-train) picks convolution algorithms at run time, and the choice can depend
  on free GPU memory, which the VRAM tier changes.
* `order=lerobot` follows `persistent_workers=False` (LeRobot's setting) and a `cycle()` that finishes each
  epoch before starting the next.
* GPU kernels themselves are not deterministic by default (`cudnn.benchmark`, atomics), in LeRobot too.
* The cache uses `index` = absolute frame number, which holds for datasets written by LeRobot.

## License

Apache-2.0. `fastload.train` wraps `lerobot.scripts.lerobot_train` from
[LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0, © The HuggingFace Inc. team).
