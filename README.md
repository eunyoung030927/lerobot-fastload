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

**This branch (`lerobot-0.5`) is for LeRobot 0.5.x** (tested with 0.5.1 on CPU; Python >= 3.12).

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
pip install git+https://github.com/eunyoung030927/lerobot-fastload.git@lerobot-0.4
# or, to edit / run the tests and benchmarks:
git clone -b lerobot-0.4 https://github.com/eunyoung030927/lerobot-fastload.git
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
* Size limit: `--max_cache_gb` (default 300). A larger dataset stores whole episodes, spread over the dataset, up to
  that size, and training decodes the frames of the other episodes as it goes ([Cache size limit](#cache-size-limit));
  `-1` = no limit.
* Resumable: rerun the same command after an interruption. `--verify N` re-checks N random frames
  against LeRobot bit-exactly (default 32).
* Disk check: before writing, `free disk - bytes still to write` must stay at least `--min_free_disk_gb`
  (default 20), otherwise it stops with the cache size, the free space and the options (another disk, fewer
  cameras with `--keys`, plain `lerobot-train`). This is a safety margin kept free for checkpoints and logs, not a
  limit on the cache size; `-1` turns it off.
* The cache is tied to the dataset's `meta/info.json`; a changed dataset needs a rebuild.

This step can also be left to the training run (`--fastload.cache_mode`, see [Cache modes](#cache-modes)).

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
| `--fastload.cache_dir` | required | cache built in step 1 (or by the training run, see `cache_mode`) |
| `--fastload.cache_mode` | `prebuilt` | `prebuilt` = the cache must be complete; `build_first` = build it before training starts; `during_training` = the first epoch decodes the frames it needs into the cache ([Cache modes](#cache-modes)) |
| `--fastload.decode_workers` | see text | decoder processes for `build_first` (default half the CPU cores) / `during_training` (default half the CPU cores divided by the training processes of the node) |
| `--fastload.background_fill` | true | `during_training`: while decoders are idle, also decode frames no batch asked for yet, so the cache completes |
| `--fastload.min_free_disk_gb` | 20 | `build_first` / `during_training`: free disk that must remain after the cache is written (safety margin, not a size limit; -1 = off) |
| `--fastload.max_cache_gb` | 300 | `build_first` / `during_training`: largest cache they create; a bigger dataset stores whole episodes up to this size and the other frames are decoded during training ([Cache size limit](#cache-size-limit); -1 = no limit) |
| `--fastload.delete_cache` | true | delete the cache when training ends normally, if a training run created it (`build_first` / `during_training`); a `build_cache` cache is never deleted ([Deleting the cache](#deleting-the-cache-after-training)) |
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

## Cache modes

`--fastload.cache_mode` decides where the cache comes from. The default stays `prebuilt`.

| mode | what happens | use it when |
|---|---|---|
| `prebuilt` (default) | the cache from step 1 must be complete; otherwise training stops at once with an error that names the other two modes and the `build_cache` command | the cache is built ahead (one build, many runs, several nodes copying one cache) |
| `build_first` | a missing or incomplete cache is built (resumably, by `build_cache`'s code) before training starts, then training runs on the complete cache | you want one command, and the dataset is decoded faster in bulk than training would read it |
| `during_training` | no separate step: the cache files are allocated, and FastLoader's first epoch decodes the frames its batches need (in decoder processes) and stores them; later epochs read the cache | you want training to start right away; the first epoch runs at about video-decoding speed, the rest at cache speed |

All three give LeRobot's batches, order, RNG states and augmentation exactly as with a prebuilt cache
(`tests/test_cache_modes.py`), and all three store the same bytes (`build_first` and `during_training` decode with
`build_cache`'s worker code through LeRobot's own reader).

**`build_first`** runs before `lerobot_train.train()`, i.e. before the Accelerator and the process group exist, so a
long build cannot run into an NCCL / accelerate timeout or one of LeRobot's barriers. The dataset is taken from the
`--dataset.repo_id`, `--dataset.root`, `--dataset.revision`, `--dataset.video_backend` and `--tolerance_s` arguments,
or, when resuming with `--config_path=.../train_config.json`, from that file (command-line values win).

**`during_training`** in detail:

* Storage: the frame files are allocated at full size up front, plus `filled.npy` (one byte per frame). A decoder
  writes a frame, flushes it, and only then sets its byte, so nothing reads a frame before it is complete. A
  stopped or crashed run leaves the filled frames in place and the next run decodes only the rest. When every frame
  is filled, `COMPLETE` is written (atomic rename) and the directory is an ordinary prebuilt cache.
* Decoding: a pool of `--fastload.decode_workers` processes. The prefetch thread hands it the frames of the next
  batches that are not cached yet (observation history included), at least 8 batches ahead, and waits before
  assembling a batch until its frames are filled (at least 4 batches are prepared ahead in this mode). While
  decoders are idle they decode the remaining frames in order (`--fastload.background_fill`), so the cache
  completes even for frames training never reads (e.g. LeRobot's dropped last frames of each episode).
* Exactness: decoding happens in other processes, so the training process's RNG is untouched, and frames are read
  only once stored; batches and RNG states are those of a prebuilt run. `augment_device` and decode-hook wrappers
  behave as with a prebuilt cache (decoder processes undo a wrapper so the cache holds the decoder's own frames, as
  `build_cache` requires).
* VRAM tier: not built while frames are missing; it is built (sized from the memory measured then) once the cache
  is complete, and `memory_guard` works as usual from then on.
* Needs `--fastload.loader=fast` (FastLoader does the filling). Indexing a `CachedLeRobotDataset` whose frames are
  not filled yet decodes them with LeRobot's own code.

**Several processes** (`accelerate launch` / `torchrun`): the cache directory is node-local. On each node only
`LOCAL_RANK` 0 builds (`build_first`) or allocates (`during_training`) the cache; the other local ranks log that they
wait, poll for it, and stop with the leader's error if it failed (the leader leaves a marker in the temp directory).
With `during_training` all ranks of a node fill the same files: each rank's decoders write the frames its batches
need (a frame written by two ranks gets identical bytes), any rank sets `filled` bytes after its own writes, and
`COMPLETE` is written by an atomic rename. Ranks on different nodes each fill their own local copy. Do not point
ranks of different nodes at one shared directory.

**Disk check** (`--fastload.min_free_disk_gb`, `build_cache --min_free_disk_gb`, default 20): before anything is
written, `free disk - bytes still to write` must stay at least this large (a resumed or partly filled cache only
counts what is not on disk yet). Otherwise the run stops, before creating any cache file, with the cache size, the
free space and the options: put the cache on another disk (`--fastload.cache_dir`), cache fewer cameras
(`build_cache --keys`, then `--fastload.loader=dataloader` decodes the others), or train with plain `lerobot-train`.
It is a safety margin kept free for checkpoints, logs and the system, not a limit on the cache size.

## Cache size limit

`--fastload.max_cache_gb` / `build_cache --max_cache_gb` (default 300 GB; `-1` = no limit) caps the frames a cache
holds. If the dataset's frames take more, the cache stores whole episodes, spread evenly over the dataset (not just
the first ones), up to that size, and records them in `stored.npy`. The frames of the other episodes are decoded
while training runs:

* FastLoader: by its decoder processes (`--fastload.decode_workers`), a few batches ahead, into a small scratch area
  `<cache_dir>/.transient/<host>-<pid>/` (a few GB at most, rows reused as batches are assembled, removed at the end),
  then gathered with the cached frames. With `during_training` the same processes also fill the stored episodes.
* `--fastload.loader=dataloader` (LeRobot's DataLoader over the cache): by LeRobot's own decoding in its workers.

Batches, order and RNG states are LeRobot's either way (`tests/test_cache_modes.py`: `limited`, and identical ACT
losses with half the frames cached). Speed lies between plain LeRobot and a full cache: each batch still waits for
its uncached frames, so the gain grows with the cached share (decoding work per batch drops to the uncached share).
The VRAM tier holds stored frames only.

The limit is fixed when the cache is created; a resumed or reused cache keeps its episodes (a warning names the
difference). The frame files have the full apparent size but are sparse: frames not stored take no disk space (`du`
shows the real size; copy such a cache with `cp --sparse=always` or `rsync -S`). The disk check counts only the
stored frames.

## Deleting the cache after training

A cache created by a training run (`build_first` / `during_training` on a missing cache; marked by a
`CREATED_BY_TRAINING` file) is deleted when `fastload.train` ends normally (`--fastload.delete_cache`, default true).

* Never deleted: a cache built with `python -m fastload.build_cache` (reused across runs), a cache any other training
  process still uses (each process holds a lease file in `<cache_dir>/.users/`; the last one to finish deletes; leases
  of dead processes on the same host are ignored), and a run that failed or was stopped (an OOM stop for
  `fastload.supervise` included), so a resumed run reuses the cache.
* `--fastload.delete_cache=false` keeps it, e.g. for several runs on the same data.
* The log says `fastload: deleted the cache <dir> (... GB on disk)`.

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

`make_loader(..., cache_mode="during_training")` fills a missing cache while it serves batches (`"build_first"`
builds it first); with several processes call it from every rank, as above.

`dataset` is your `LeRobotDataset` (delta timestamps and image transforms are taken from it).
`fastload.CachedLeRobotDataset.from_dataset(dataset, cache_dir)` alone gives a drop-in dataset for an
ordinary `DataLoader` (any `num_workers`) if you only want to skip video decoding.

## Benchmark and test on your data

```bash
python tests/test_cache_modes.py        # cache modes on synthetic data (CPU)
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

Training scripts sometimes change frames right after decoding by wrapping the `decode_video_frames` that
LeRobot's reader calls (LeRobot 0.5: `lerobot.datasets.dataset_reader.decode_video_frames`) (e.g. resizing 480x640 to 240x320 in the DataLoader
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
