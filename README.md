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

* **Same batches as LeRobot.** Frames are decoded by LeRobot's own `_query_videos` and stored as the
  decoder's uint8 output, so cached frames are bit-identical. Action chunks, `*_is_pad`, observation history,
  task strings and image transforms follow `LeRobotDataset.__getitems__`. `tests/test_equivalence.py`
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

**This branch (`lerobot-0.7`) is for LeRobot 0.7.x** (tested with 0.7.0 on CPU; Python >= 3.12).

What changed in LeRobot 0.7 and how this branch follows it:

* `lerobot_train` builds its DataLoaders in `make_dataloaders` and passes `generator=torch.Generator().manual_seed(seed)`:
  `base_seed` is drawn from that generator, not from the global RNG. FastLoader takes the same `generator` and
  its index DataLoader draws from it (with persistent workers only on the first epoch, as before). Everything
  else in the loader setup (`EpisodeAwareSampler(seed, epoch)` for every map-style dataset, `persistent_workers`,
  `prefetch_factor`, `spawn`, `return_uint8=True` datasets, uint8 -> float on the device in the loop) is as in 0.6.
* Image transforms were rewritten: `ImageTransforms` is an `nn.Module` that runs the batched transforms
  (`BatchedRandomSubsetApply` over `BatchedTransform`s, whose random draws all happen in `make_params`) on a
  batch of one, and `image_transforms.backend` selects where they run:
  * `dataloader` (default): per sample in the DataLoader workers. fastload draws the parameters exactly like
    LeRobot (same calls, same order, same shapes, from the emulated worker RNG streams), including the draws a
    torchvision transform wrapped in `PerSampleTransform` (`JPEGCompression`, any `torchvision.transforms.v2`
    class by name) makes while it runs, and applies LeRobot's own transform math with them, with LeRobot's uint8
    handling (`/255`, transform, `round` back to uint8). `--fastload.augment_device=cpu` stays bit-identical.
  * `gpu`: the dataset has no transforms and `lerobot_train` augments each batch in the training loop
    (`ImageAugmentationProcessorStep`, own generator). FastLoader then just delivers the unaugmented batch.
* Batches come from `LeRobotDataset.__getitems__` (`DatasetReader.get_items`): same values and RNG draws as per-item
  `__getitem__`, but video frames now come after the tabular columns and `*_is_pad` in each item. FastLoader
  produces LeRobot's key order.
* `lerobot_collate_fn` now keeps the language columns (`language_persistent` / `language_events`) as per-sample
  Python lists (policy processors render them). FastLoader reads them from the parquet rows of each batch.
* `update_policy` runs inside `accelerator.accumulate(policy)` and steps an optional EMA, so the one-process OOM
  retry only redoes a step when that is exact (see "Running out of GPU memory").
* Non-default storage formats (`meta.storage_format`, e.g. `lance`) and `--dataset.streaming=true` are refused with
  an error (there is nothing to cache in the default layout; train those with plain `lerobot-train`).
* Context parallelism (`--parallelism.context_parallel.*`) is rejected by LeRobot 0.7.0 itself. `order=lerobot`
  shards like LeRobot because accelerate prepares the index DataLoader with the same device mesh; `order=shard`
  refuses context / tensor parallelism.

## Install

fastload hooks into LeRobot internals that change between minor releases, so there is one branch per
LeRobot minor version. Install the branch that matches the LeRobot in your environment:

| LeRobot | branch |
|---|---|
| 0.4.x (>=0.4.2) | `lerobot-0.4` |
| 0.5.x | `lerobot-0.5` |
| 0.6.x | `lerobot-0.6` |
| 0.7.x | `lerobot-0.7` (also `main`) |

```bash
pip install git+https://github.com/eunyoung030927/lerobot-fastload.git@lerobot-0.7
# or, to edit / run the tests and benchmarks:
git clone -b lerobot-0.7 https://github.com/eunyoung030927/lerobot-fastload.git
cd lerobot-fastload && pip install -e .
```

A branch refuses to run with another LeRobot version and prints the branch to install instead.

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
* Depth cameras are skipped with a warning (12-bit / float frames are not cached).

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

With `loader=fast` no worker processes are started: `num_workers` and `persistent_workers` only select the
emulated worker RNG streams for image augmentation, and `pin_memory` / `prefetch_factor` /
`multiprocessing_context` are ignored.

## Custom training scripts

```python
import fastload

# after creating the Accelerator (rank / world size / device are read from it);
# same arguments as the DataLoader it replaces, including num_workers / persistent_workers / collate_fn / generator
dataloader = fastload.make_loader(dataset, batch_size, cache_dir="/nvme/fastload/<name>",
                                  sampler=sampler, shuffle=sampler is None, num_workers=4,
                                  persistent_workers=True, generator=torch.Generator().manual_seed(seed))
policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(policy, optimizer, dataloader, lr_scheduler)
for batch in dataloader:   # one epoch of this rank's batches; tensors already on the GPU
    ...
```

`dataset` is your `LeRobotDataset` (delta timestamps, image transforms and `return_uint8` are taken from it;
with `return_uint8=True` video cameras arrive as uint8 and your loop divides by 255, as in `lerobot_train`).
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
| sampler (`EpisodeAwareSampler` seeded from `(seed, epoch)`, or `RandomSampler`), accelerate batch sharding and RNG sync | the same objects, iterated over row indices (`accelerator.prepare` prepares them) |
| `base_seed` (and a `RandomSampler`'s seed) drawn from the DataLoader's `generator` (`lerobot_train` 0.7: `torch.Generator().manual_seed(seed)`; without one, the process RNG) when an epoch starts; with `persistent_workers` only the first epoch draws `base_seed` | the same draws, from the same generator, at the same point |
| worker `w` reseeds `torch` / `numpy` / `random` from `base_seed + w` when it starts (persistent workers: once); batch `k` of every epoch is fetched by worker `k % num_workers` | the same streams, emulated in the main thread (no worker processes) |
| `num_workers=0`: transforms draw from the process RNG, one batch ahead (accelerate prefetches) | the same |
| `image_transforms` (`backend=dataloader`) per sample and camera on the CPU, on uint8 video frames (`return_uint8`): `ImageTransforms` draws every parameter in `make_params` (plus the draws of wrapped torchvision transforms while they run) | same parameters, drawn the same way on the CPU; LeRobot's transform math applied on the GPU (float rounding, see Limitations) or, with `--fastload.augment_device=cpu`, LeRobot's CPU path (bit-identical) |
| `backend=gpu`: no dataset transforms; `lerobot_train` augments the batch in the loop with its own generator | the same (FastLoader delivers the unaugmented batch) |

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
2. **OOM in a step, one process**: halve the tier and retry the step, if the retry is exactly the step LeRobot
   runs: no gradient accumulation (`--accelerator.gradient_accumulation.steps=1`; a retry would lose the gradients
   of earlier micro-batches) and the OOM came before the optimizer step (so the parameters, optimizer, scaler,
   scheduler and EMA are untouched). The process RNG states and the accelerator's step counter are restored before
   the retry, so the retried step draws the same noise. Otherwise the process stops like several processes do
   (below), for `fastload.supervise` to resume from the last checkpoint.
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

Training scripts sometimes change frames right after decoding by wrapping the `decode_video_frames` that
LeRobot's reader calls (LeRobot 0.6 and 0.7: `lerobot.datasets.dataset_reader.decode_video_frames`) (e.g. resizing
480x640 to 240x320 in the DataLoader workers). fastload never decodes video, so it calls that wrapper itself,
with the reader's arguments (including `return_uint8=` and `is_depth=False`), while LeRobot's backend decoders
(`decode_video_frames_torchcodec` / `_pyav`) answer with the cached frames: the wrapper's own code runs on
exactly the frames LeRobot's decoder would return (uint8 under `lerobot_train`, which sets `return_uint8=True`;
a wrapper written for 0.5's float frames has to handle uint8), at the same point (before `image_transforms`).
With `--fastload.decode_hook_device=cpu` (default) the result is bit-identical to LeRobot; those cameras then
skip the VRAM tier. `gpu` runs the wrapper on the batch on the GPU instead (the wrapper must accept CUDA
tensors; e.g. bilinear `interpolate` has no uint8 CUDA kernel). Build the cache in a process without the wrapper (`build_cache` refuses
otherwise).

## Limitations

* Single `LeRobotDataset` in LeRobot's default storage format (as `lerobot-train`); `StreamingLeRobotDataset`
  (`--dataset.streaming=true`), `MultiLeRobotDataset` and other storage formats (`meta.storage_format`, e.g.
  `lance`) are refused.
* Image transforms must be LeRobot's `ImageTransforms` (what `lerobot_train` builds) or a plain
  `torchvision.transforms.v2` pipeline (custom scripts); the deprecated per-transform front ends of
  `lerobot.transforms` (`RandomSubsetApply`, `GaussianNoise`, ... used directly) are refused. Their parameters match
  LeRobot's; with the default `augment_device=gpu` augmented pixels can differ from LeRobot's CPU result by float
  rounding: ~1e-6 on float32 cameras, and on uint8 video frames (`return_uint8`) the rounding back to uint8 can land
  one level (1/255) apart at some pixels. On CUDA the batched transforms run on the whole batch at once (as on
  LeRobot's `gpu` backend); per-sample torchvision transforms (`JPEGCompression`, v2 classes by name) run sample by
  sample. Random parameters, including GaussianNoise's noise, are drawn on the CPU like LeRobot and copied.
  `augment_device=cpu` makes everything bit-identical at the cost of the VRAM tier and CPU time. A per-sample
  torchvision transform that changes the frame size is applied, but later per-sample transforms draw their
  parameters for the original size.
* Depth cameras (`meta.depth_keys`, 12-bit / float frames) are not cached: `build_cache` skips them,
  `CachedLeRobotDataset` (`--fastload.loader=dataloader`) decodes them with LeRobot's code and reads the RGB
  cameras from the cache, and FastLoader refuses datasets with depth cameras.
* Language columns (`language_persistent` / `language_events`) are supported the way `lerobot_train` uses them:
  `lerobot_collate_fn` keeps them as per-sample lists, and FastLoader requires that `collate_fn` for such datasets
  (any other `collate_fn` is refused).
* `cudnn.benchmark` (on in lerobot-train) picks convolution algorithms at run time, and the choice can depend
  on free GPU memory, which the VRAM tier changes.
* `order=lerobot` follows the DataLoader's `persistent_workers` (pass it to `make_loader` in custom scripts;
  `fastload.train` takes it from `lerobot_train`) and a `cycle()` that finishes each epoch before starting the
  next. With persistent workers it relies on PyTorch restarting the worker cycle at worker 0 every epoch
  (checked against torch 2.11 and 2.15).
* GPU kernels themselves are not deterministic by default (`cudnn.benchmark`, atomics), in LeRobot too.
* The cache uses `index` = absolute frame number, which holds for datasets written by LeRobot.

## License

Apache-2.0. `fastload.train` wraps `lerobot.scripts.lerobot_train` from
[LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0, © The HuggingFace Inc. team).
