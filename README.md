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
* **Adapts to each run.** Nothing to tune per GPU count: each rank takes whole episodes (spread over
  tasks), measures its memory after a few steps, and moves as much of its shard to VRAM as fits.
  Frames not in VRAM are gathered from the cache by a background thread and copied on a side CUDA stream.

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
| `--fastload.loader` | `fast` | `fast` = FastLoader; `dataloader` = LeRobot's DataLoader reading the cache (no VRAM tier) |
| `--fastload.placement` | `auto` | `auto` grows a VRAM tier after warm-up; `gpu` puts the whole shard in VRAM at start; `host` never uses VRAM |
| `--fastload.warmup_steps` | 20 | steps before measuring memory (`auto`) |
| `--fastload.gpu_margin_gb` | max(2 GB, 5%) | VRAM kept free above the measured peak |
| `--fastload.max_gpu_cache_gb` | none | cap on the VRAM tier |
| `--fastload.prefetch` | 2 | batches prepared ahead |
| `--fastload.oom_retry` | true | single process only: on CUDA OOM halve the VRAM tier and retry the step |

`num_workers`, `pin_memory` and `prefetch_factor` are ignored with `loader=fast`.

## Custom training scripts

```python
import fastload

# after creating the Accelerator (rank / world size / device are read from it)
policy, optimizer, lr_scheduler = accelerator.prepare(policy, optimizer, lr_scheduler)  # no dataloader here
dataloader = fastload.make_loader(dataset, batch_size, cache_dir="/nvme/fastload/<name>",
                                  sampler=sampler, shuffle=True)
for batch in dataloader:   # one epoch of this rank's shard; tensors already on the GPU
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

## How sharding works with DDP

Each rank owns whole episodes (round-robin after sorting episodes by task), shuffles within its shard
every epoch, and draws `batch_size` samples per step. Every optimizer step therefore averages
gradients over samples from all shards, which is the same per-step mixing a distributed sampler gives.
Shards are fixed for the run; with `world_size=1` the shard is the whole dataset.

## Limitations

* Single `LeRobotDataset` (as `lerobot-train`); `StreamingLeRobotDataset` is not supported.
* Image transforms run per sample on the GPU (same semantics as LeRobot, different RNG stream).
* OOM recovery is single-process only; under DDP keep `gpu_margin_gb` conservative.
* The cache uses `index` = absolute frame number, which holds for datasets written by LeRobot.

## License

Apache-2.0. `fastload.train` wraps `lerobot.scripts.lerobot_train` from
[LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0, © The HuggingFace Inc. team).
