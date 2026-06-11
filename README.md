## TAROS / ROS 2

When used as a submodule in `taros_autonomy_ws` (`src/terrain_toolkit`):

```bash
# One-time external dependency (public NVIDIA wheel — not this package)
pip install 'warp-lang>=1.12.1'

# From workspace root
./rebuild_specific.sh terrain_toolkit_ros
source install/local_setup.bash
ros2 launch terrain_toolkit_ros taros.launch.py
```

The library is installed by colcon into the workspace overlay — **do not** `pip install` this repo.

## Requirements

- Python ≥ 3.12 (ROS 2 Kilted)
- NVIDIA GPU with CUDA support
- `python3-numpy` (from ROS / system)
- `warp-lang` (install once via pip)

## Build (colcon)

Clone into a colcon workspace as `src/terrain_toolkit`, then:

```bash
cd <ws>
colcon build --packages-select terrain_toolkit_ros --symlink-install
source install/local_setup.bash
```

## Offline / library usage

After sourcing the workspace overlay, Python scripts at the repo root can import the library:

```bash
source install/local_setup.bash
python3 example.py
python3 test_synthetic.py --preset noisy
python3 test_ouster.py --path ouster.npy
python3 profile_pipeline.py --path ouster.npy
```

Optional dev plotting deps: `pip install matplotlib plotly`

# Terrain Toolkit

GPU-accelerated point cloud → heightmap → traversability cost map using
[NVIDIA Warp](https://github.com/NVIDIA/warp).

Full pipeline on a 68k-point Ouster frame runs in **~5.5 ms / frame** on an
RTX A500 laptop GPU (≈180 FPS).

**📖 Full documentation:** [aleskucera.github.io/terrain_toolkit](https://aleskucera.github.io/terrain_toolkit/)

## What's in the box

| Stage | Module | Purpose |
|---|---|---|
| Outlier removal | `outlier/` | `StatisticalOutlierFilter` (SOR) or `RadiusOutlierFilter` (ROR), GPU-native k-NN via `wp.HashGrid` |
| Heightmap raster | `heightmap/` | `HeightMapBuilder` — max/mean/min/count layers in one kernel pass |
| Hole filling | `heightmap/` | `multigrid_inpaint` — NaN-aware Laplace diffusion on a pyramid |
| Smoothing | `heightmap/` | `gaussian_smooth` — NaN-aware separable blur |
| Geometric cost | `traversability/` | `GeometricTraversabilityAnalyzer` — slope + signed step + roughness |
| Post-process | `traversability/` | `ObstacleInflator`, `TemporalGate`, `SupportRatioMask` |
| Orchestration | `pipeline.py` | `TerrainPipeline` — points in, `TerrainMap` out |
| ICP | `icp/` | `IcpAligner` — GPU-native point-to-point ICP (standalone, not in the pipeline) |

## Quick start

```python
import numpy as np
from terrain_toolkit import (
    TerrainPipeline, TraversabilityConfig, FilterConfig,
    RadiusOutlierFilterConfig,
)

pipe = TerrainPipeline(
    resolution=0.1,
    bounds=(-5, 5, -5, 5),
    outlier=RadiusOutlierFilterConfig(),       # fast, radius-based
    traversability=TraversabilityConfig(),
    filter=FilterConfig(),
    layers=("traversability",),                # only download what you need
)

tm = pipe.process(points)                      # points: (N, 3) float32
cost = tm.traversability                       # (H, W) float32, NaN = unknown
```

See [`example.py`](example.py) for a fully-explicit configuration and
[`profile_pipeline.py`](profile_pipeline.py) for a per-stage timing harness.

## Deeper docs

- [**Pipeline reference**](docs/pipeline.md) — `TerrainPipeline`, `TerrainMap`,
  selective download, layer semantics
- [**Outlier filtering**](docs/outlier.md) — SOR vs ROR, config, which to pick
- [**Heightmap building blocks**](docs/heightmap.md) — builder, inpaint, smooth
- [**Traversability**](docs/traversability.md) — cost layers, filter chain,
  tuning guide
- [**Performance**](docs/performance.md) — current per-stage profile,
  optimization history, how to measure

## Test scripts

See [Offline / library usage](#offline--library-usage) above.
