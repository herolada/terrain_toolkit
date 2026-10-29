# Performance

## Current profile

Measured on an **RTX A500 (4 GiB, sm_86)** with a real Ouster frame
(`ouster.npy`, 68,712 points, 81×158 grid at 0.1 m/cell). Defaults throughout
(ROR outlier, full traversability stack, all layers downloaded).

| Stage | ms (median) | % of frame |
|---|---:|---:|
| upload | 0.22 | 3.9% |
| **outlier (ROR)** | **2.90** | **52.2%** |
| heightmap build | 0.14 | 2.5% |
| inpaint | 1.27 | 22.8% |
| smooth | 0.11 | 2.0% |
| analyzer | 0.09 | 1.7% |
| inflate | 0.11 | 2.0% |
| temporal gate | 0.08 | 1.4% |
| support mask | 0.05 | 0.9% |
| download | 0.58 | 10.4% |
| **total** | **5.55** | **100%** |

**≈ 180 FPS** on laptop-grade hardware.

Reproduce with `python profile_pipeline.py --path ouster.npy`. Each stage is
bracketed by `wp.synchronize()` so the per-stage GPU work is attributed
correctly.

## How to measure

```bash
# default (ROR outlier)
python profile_pipeline.py --path ouster.npy

# use SOR instead for comparison
python profile_pipeline.py --path ouster.npy --sor

# skip outlier filtering
python profile_pipeline.py --path ouster.npy --no-outlier

# tune iteration count / warmup
python profile_pipeline.py --path ouster.npy --frames 200 --warmup 20
```

The profiler reports the **median** of N frames after the warmup runs.

## Selective download

Restricting `layers=` lowers the download and the total:

| `layers` selector | total (ms) |
|---|---:|
| `None` (all 9 layers) | 5.34 |
| `("traversability", "elevation")` | 4.88 |
| `("traversability",)` | 4.82 |

The compute is identical; only D2H copies change. Useful if you're publishing
only the final cost map to a consumer.

## What dominates

After the current round of optimizations:

1. **Outlier (52%)** — hashgrid build + fused k-NN kernel. ROR's early-exit
   already caps the per-point work; remaining cost is the grid build and the
   single kernel launch itself. Nothing obvious left without changing
   algorithms (voxel downsample first, range-image filter, …).
2. **Inpaint (23%)** — dense Jacobi over a 5-level pyramid, ~800 kernel
   invocations per frame. Mostly wasted work on cells that aren't holes.
   Sparse iteration or a push-pull pyramid would replace this; not yet
   implemented.
3. **Download (10%)** — constant with selective download off. See above.

Everything else is < 5% and roughly at its per-launch floor — fusing these
small kernels would shave < 0.1 ms total.

## Optimization history (what's been done)

- **SOR → ROR for the default outlier path** (#6×)
  Early-exit at `min_neighbors` — 16.5 ms → 2.9 ms on this dataset.
- **Fused atomic reduction in SOR's mean-distance kernel**
  One kernel launch fewer; ~0.5 ms saved on the SOR path.
- **Lazy hashgrid + optional `bounds`**
  Skipped the per-frame `points.numpy()` readback that used to size the grid.
  Optional `bounds=...` skips even the first-call readback entirely.
- **Selective download**
  `TerrainMap` fields are now all optional; only selected ones are copied to CPU.
- **Signed step-height**
  No speed impact, but quality improvement: drops and bumps are now scored
  independently with their own saturation thresholds.
- **Host CPU load** (matters on Jetson, where the CPU — not the GPU — was the
  bottleneck: the grid is tiny, so frames are launch- and wait-bound)
  - Host waits use a `CU_EVENT_BLOCKING_SYNC` event (`cuda_sync.wait`) instead
    of `wp.synchronize()`, which busy-spins a core for the whole GPU time.
  - Inpaint → smooth → costs → inflation → obstacle count run from one CUDA
    graph, captured once on the second frame (was: a fresh pyramid plus one
    graph capture/instantiate per pyramid level every frame, ~250 launches).
  - The inpaint fixed-mask is a kernel (was a D2H + numpy + H2D round-trip).
  - Points go up through a pinned staging buffer; the optional sensor → grid
    `transform`, NaN removal and `z_max` cut run on the GPU. Layers come down
    with async copies into pinned buffers and a single wait.
  - ROS node: zero-copy x/y/z view of the PointCloud2, no float64 numpy
    transform (its multithreaded OpenBLAS call left worker threads spinning
    on several cores — ~3 extra cores busy on an 8-core laptop),
    `OPENBLAS_NUM_THREADS=1`, queue depth 1, `array('b')` for the
    OccupancyGrid data.

  On an MX250 laptop (200×200 grid, Ouster-like 131k-point frames): host CPU
  per frame 11.6 → 5.8 ms (no outlier) and 19.9 → 6.7 ms (ROR) in the
  library, with bit-identical outputs.

## Potential future wins

In rough order of ROI:

1. **Sparse inpaint** — iterate only hole cells, or switch to a push-pull
   (normalized-convolution) pyramid. Expected 0.5–1 ms on typical lidar data.
2. **Overlap upload / compute / download with streams** — halves steady-state
   latency when streaming.
3. **Kernel fusion across analyzer stages** — slope + step + roughness all
   read overlapping 3×3 to 5×5 elevation windows; fusing saves launches and
   redundant loads. Marginal (~0.1 ms) at the current grid size.

## Hardware scaling

The profile above is the worst-case — an RTX A500 is a 2048-core laptop GPU
at ~3 TFLOPS. On a desktop 30/40-series card expect every stage to shrink
roughly proportionally to memory bandwidth / SM count. Kernel launch overhead
dominates at small grid sizes so the relative stage percentages may shift.
