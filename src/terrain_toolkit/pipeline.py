from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import warnings

import numpy as np
import warp as wp

from . import cuda_sync
from .heightmap import FlatGroundFootprint
from .heightmap import FootprintConfig
from .heightmap import GaussianSmoother
from .heightmap import HeightMapBuilder
from .heightmap import MultigridInpainter
from .heightmap.kernels import transform_filter_kernel
from .outlier import OutlierFilterConfig
from .outlier import RadiusOutlierFilter
from .outlier import RadiusOutlierFilterConfig
from .outlier import StatisticalOutlierFilter
from .traversability import FilterConfig
from .traversability import GeometricTraversabilityAnalyzer
from .traversability import ObstacleInflator
from .traversability import OcclusionConfig
from .traversability import OcclusionMask
from .traversability import SupportRatioMask
from .traversability import TemporalGate
from .traversability import TraversabilityConfig

PrimaryLayer = Literal["max", "mean", "min"]

LayerName = Literal[
    "max", "mean", "min", "count",
    "elevation",
    "slope_cost", "step_cost", "roughness_cost", "traversability",
]

_ALL_LAYERS: tuple[str, ...] = (
    "max", "mean", "min", "count",
    "elevation",
    "slope_cost", "step_cost", "roughness_cost", "traversability",
)
_COST_LAYERS: frozenset[str] = frozenset(
    ("slope_cost", "step_cost", "roughness_cost", "traversability"),
)


@dataclass
class TerrainMap:
    """Output of `TerrainPipeline.process`.

    Fields are populated only when their layer was selected for download (see
    `TerrainPipeline(layers=...)`) *and* the corresponding stage was enabled.
    Everything else is `None`.
    """

    resolution: float
    bounds: tuple[float, float, float, float]

    # raw reductions
    max: np.ndarray | None = None
    mean: np.ndarray | None = None
    min: np.ndarray | None = None
    count: np.ndarray | None = None

    # primary reduction after inpaint + smooth
    elevation: np.ndarray | None = None

    # geometric cost layers (populated only when traversability is enabled)
    slope_cost: np.ndarray | None = None
    step_cost: np.ndarray | None = None
    roughness_cost: np.ndarray | None = None
    traversability: np.ndarray | None = None

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return all non-None layers as a flat name → array dict."""
        d: dict[str, np.ndarray] = {}
        for name in _ALL_LAYERS:
            arr = getattr(self, name)
            if arr is not None:
                d[name] = arr
        return d


@dataclass
class TerrainMapGPU:
    """Device-resident output of `TerrainPipeline.process(return_device=True)`.

    Each field is a live `wp.array` handle into the pipeline's internal buffers —
    NOT a copy. The handles are BORROWED: they stay valid only until the next
    `process()` call on the same pipeline (buffers are reused frame-to-frame). A
    consumer that needs to retain a layer across frames must `wp.clone()` it.

    Lets a downstream Warp library consume the elevation grid on-device with no
    GPU→CPU→GPU round-trip. Grid layout matches `TerrainMap`: shape
    (height, width) = (ny, nx), row=y, col=x; cell-center world coords
    x = xmin + (j+0.5)*resolution, y = ymin + (i+0.5)*resolution.
    """

    resolution: float
    bounds: tuple[float, float, float, float]
    elevation: wp.array | None = None
    traversability: wp.array | None = None
    slope_cost: wp.array | None = None
    step_cost: wp.array | None = None
    roughness_cost: wp.array | None = None


class TerrainPipeline:
    """Points → (max, mean, min, count) → inpaint → smooth → cost → filter.

    Single entry point: `process(points)` returns a fully-populated
    `TerrainMap`. Stateful: reuses GPU buffers and filter hysteresis across
    calls, so the same instance should be reused frame-to-frame.

    Internally keeps data on the GPU: point cloud is uploaded once, every
    stage consumes and produces `wp.array`, and a single download happens
    at the end to build the numpy-backed `TerrainMap` for the caller.

    CPU cost is kept low on CUDA: all buffers are preallocated, upload and
    download go through pinned host memory, the grid stages (inpaint → smooth
    → costs → inflation) are replayed from one CUDA graph captured on the
    second frame (`use_cuda_graph=False` disables it), and every host wait
    blocks instead of busy-spinning (see `cuda_sync`).
    """

    def __init__(
        self,
        resolution: float,
        bounds: tuple[float, float, float, float],
        *,
        primary: PrimaryLayer = "max",
        inpaint: bool = True,
        smooth_sigma: float = 0.0,
        inpaint_iters_per_level: int = 50,
        inpaint_coarse_iters: int = 200,
        z_max: float | None = None,
        outlier: OutlierFilterConfig | RadiusOutlierFilterConfig | None = None,
        traversability: TraversabilityConfig | None = None,
        filter: FilterConfig | None = None,
        occlusion: OcclusionConfig | None = None,
        footprint: FootprintConfig | None = None,
        layers: tuple[str, ...] | None = None,
        device: str | wp.context.Device | None = None,
        use_cuda_graph: bool = True,
    ):
        if primary not in ("max", "mean", "min"):
            raise ValueError(f"primary must be 'max', 'mean', or 'min'; got {primary!r}")
        if traversability is not None and not inpaint:
            raise ValueError(
                "traversability requires inpaint=True — the cost kernels assume a filled grid"
            )
        if filter is not None and traversability is None:
            raise ValueError("filter is only meaningful when traversability is enabled")
        if occlusion is not None and traversability is None:
            raise ValueError("occlusion masks the cost map and requires traversability=...")

        # Resolve the Warp device once. Accepts "cpu", "cuda:0", a wp.Device, or
        # None (use Warp's current default). Outlier filtering uses wp.HashGrid
        # which is CUDA-only, so reject that combination up front.
        if device is None:
            self.device = wp.get_device()
        elif isinstance(device, str):
            self.device = wp.get_device(device)
        else:
            self.device = device
        if outlier is not None and not self.device.is_cuda:
            raise ValueError(
                "outlier filtering requires a CUDA device (wp.HashGrid is GPU-only); "
                f"got device={self.device}"
            )

        # Resolve which layers to download. `None` = everything the configured
        # stages produce. Skipping unused layers saves a D2H copy each — about
        # 0.05–0.1 ms per layer on an 80×150 grid.
        if layers is None:
            selected = set(_ALL_LAYERS)
        else:
            unknown = set(layers) - set(_ALL_LAYERS)
            if unknown:
                raise ValueError(
                    f"unknown layer names {sorted(unknown)}; valid: {_ALL_LAYERS}"
                )
            selected = set(layers)
        if traversability is None:
            selected -= _COST_LAYERS
        self._layers: set[str] = selected

        self.resolution = resolution
        self.bounds = bounds
        self.primary = primary
        self.z_max = z_max
        self.inpaint = inpaint
        self.smooth_sigma = smooth_sigma
        self.inpaint_iters_per_level = inpaint_iters_per_level
        self.inpaint_coarse_iters = inpaint_coarse_iters

        self.builder = HeightMapBuilder(
            resolution=resolution, bounds=bounds, device=self.device,
        )
        self.height = self.builder.height
        self.width = self.builder.width

        # Per-frame buffers, allocated once. The point buffers grow on demand.
        self._pts_capacity = 0
        self._pts_staging: wp.array | None = None  # pinned host (CUDA only)
        self._pts_raw: wp.array | None = None
        self._pts: wp.array | None = None
        self._host_layers: dict[str, wp.array] = {}  # pinned download targets
        with wp.ScopedDevice(self.device):
            self._pts_count = wp.zeros(1, dtype=wp.int32)
            # multigrid inpaint mutates the primary layer in place; the support
            # and occlusion masks need the original NaN-bearing copy.
            self._raw_primary = wp.zeros(self.builder.shape, dtype=wp.float32) if inpaint else None

        self.inpainter: MultigridInpainter | None = None
        if inpaint:
            self.inpainter = MultigridInpainter(
                self.height, self.width,
                iters_per_level=inpaint_iters_per_level,
                coarse_iters=inpaint_coarse_iters,
                device=self.device,
            )
        self.smoother: GaussianSmoother | None = None
        if smooth_sigma > 0.0:
            self.smoother = GaussianSmoother(
                self.height, self.width, smooth_sigma, device=self.device,
            )

        # Graph state for the grid stages: frame 1 runs eagerly (loads modules,
        # records the output buffers), frame 2 captures, later frames replay.
        self._use_graph = use_cuda_graph and self.device.is_cuda
        self._grid_graph = None
        self._grid_outputs: tuple | None = None

        self.outlier_filter: StatisticalOutlierFilter | RadiusOutlierFilter | None = None
        if isinstance(outlier, RadiusOutlierFilterConfig):
            self.outlier_filter = RadiusOutlierFilter(config=outlier, device=self.device)
        elif isinstance(outlier, OutlierFilterConfig):
            self.outlier_filter = StatisticalOutlierFilter(config=outlier, device=self.device)

        self.footprint_stamp: FlatGroundFootprint | None = None
        if footprint is not None:
            self.footprint_stamp = FlatGroundFootprint(
                resolution=resolution,
                bounds=bounds,
                height=self.height,
                width=self.width,
                config=footprint,
                device=self.device,
            )

        self.analyzer: GeometricTraversabilityAnalyzer | None = None
        if traversability is not None:
            self.analyzer = GeometricTraversabilityAnalyzer(
                resolution=resolution,
                height=self.height,
                width=self.width,
                config=traversability,
                device=self.device,
            )

        self.inflator: ObstacleInflator | None = None
        self.temporal_gate: TemporalGate | None = None
        self.support_mask: SupportRatioMask | None = None
        if filter is not None:
            self.inflator = ObstacleInflator(
                resolution=resolution,
                height=self.height,
                width=self.width,
                config=filter,
                device=self.device,
            )
            self.temporal_gate = TemporalGate(config=filter, device=self.device)
            self.support_mask = SupportRatioMask(
                resolution=resolution,
                height=self.height,
                width=self.width,
                config=filter,
                device=self.device,
            )

        self.occlusion_mask: OcclusionMask | None = None
        if occlusion is not None:
            self.occlusion_mask = OcclusionMask(
                resolution=resolution,
                bounds=bounds,
                height=self.height,
                width=self.width,
                config=occlusion,
                device=self.device,
            )

    def process(
        self,
        points: np.ndarray,
        *,
        transform: np.ndarray | None = None,
        footprint_plane: tuple[float, float, float] | None = None,
        return_device: bool = False,
    ) -> TerrainMap | TerrainMapGPU:
        """Run the pipeline on a point cloud.

        `points` is any (N, 3) float array (strided views are fine). Rows with
        non-finite coordinates are dropped. If `transform` (4×4, source → grid
        frame) is given, it is applied on the GPU before anything else, so the
        caller can pass raw sensor-frame points.

        Default: download the selected layers to a numpy-backed `TerrainMap`.
        `return_device=True`: skip the host copy and return a `TerrainMapGPU` of
        live device `wp.array` handles (valid only until the next `process()` —
        `wp.clone()` to retain) for a zero-copy handoff to another Warp library.
        """
        # Scope every allocation and kernel launch in this frame to the chosen
        # device — without this, helpers that call wp.array(...) or wp.zeros(...)
        # without an explicit device= argument would land on Warp's global
        # default, which can differ from self.device.
        with wp.ScopedDevice(self.device):
            dev = self._compute(points, transform, footprint_plane)
            if return_device:
                cuda_sync.wait(self.device)
                return self._wrap_device(dev)
            return self._download(dev)

    def _upload(self, points: np.ndarray, transform: np.ndarray | None) -> wp.array:
        """Upload, transform to the grid frame, and drop non-finite / z > z_max
        points, all on the device. Returns a view of the surviving points."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3); got {points.shape}")
        n = len(points)
        if n > self._pts_capacity:
            cap = max(n, int(self._pts_capacity * 1.25))
            self._pts_raw = wp.empty(cap, dtype=wp.vec3)
            self._pts = wp.empty(cap, dtype=wp.vec3)
            if self.device.is_cuda:
                self._pts_staging = wp.empty(cap, dtype=wp.vec3, device="cpu", pinned=True)
            self._pts_capacity = cap

        if self.device.is_cuda:
            # One host-side copy (with dtype conversion) into pinned memory,
            # then an async DMA to the device.
            self._pts_staging.numpy()[:n] = points
            if n > 0:
                wp.copy(self._pts_raw, self._pts_staging, count=n)
        else:
            self._pts_raw.numpy()[:n] = points

        if transform is None:
            rot, trans = wp.mat33(np.eye(3, dtype=np.float32)), wp.vec3(0.0, 0.0, 0.0)
        else:
            t = np.asarray(transform, dtype=np.float32)
            rot, trans = wp.mat33(*t[:3, :3].ravel()), wp.vec3(*t[:3, 3])
        z_max = float(self.z_max) if self.z_max is not None else float("inf")

        self._pts_count.zero_()
        wp.launch(
            transform_filter_kernel,
            dim=n,
            inputs=[self._pts_raw, rot, trans, z_max],
            outputs=[self._pts, self._pts_count],
        )
        cuda_sync.wait(self.device)
        return self._pts[: int(self._pts_count.numpy()[0])]

    def _grid_stages(self, primary_layer: wp.array) -> tuple:
        """Inpaint → smooth → costs → inflation → obstacle count.

        Only kernel launches and device copies into preallocated buffers, with
        fixed pointers and launch dims frame to frame — that is what lets it be
        captured once into a CUDA graph and replayed.
        """
        raw_primary = primary_layer
        elevation = primary_layer
        if self.inpainter is not None:
            wp.copy(self._raw_primary, primary_layer)
            raw_primary = self._raw_primary
            elevation = self.inpainter.run(primary_layer)
        if self.smoother is not None:
            elevation = self.smoother.run(elevation)

        costs = None
        inflated = None
        if self.analyzer is not None:
            costs = self.analyzer.compute(elevation)
            if self.inflator is not None:
                inflated = self.inflator.apply(costs.total)
                self.temporal_gate.count(inflated)
        return raw_primary, elevation, costs, inflated

    def _run_grid_stages(self, primary_layer: wp.array) -> tuple:
        if self._grid_graph is not None:
            wp.capture_launch(self._grid_graph)
            return self._grid_outputs
        if self._use_graph and self._grid_outputs is not None:
            try:
                with wp.ScopedCapture(device=self.device) as capture:
                    outputs = self._grid_stages(primary_layer)
            except Exception as exc:  # fall back to eager launches for good
                warnings.warn(f"CUDA graph capture failed, running eagerly: {exc}")
                self._use_graph = False
            else:
                self._grid_graph = capture.graph
                self._grid_outputs = outputs
                wp.capture_launch(self._grid_graph)
                return outputs
        self._grid_outputs = self._grid_stages(primary_layer)
        return self._grid_outputs

    def _compute(
        self,
        points: np.ndarray,
        transform: np.ndarray | None = None,
        footprint_plane: tuple[float, float, float] | None = None,
    ) -> dict[str, object]:
        """Run all GPU stages and return the device buffers (no download)."""
        # Single upload to the active device — every stage below consumes wp.array.
        pts_wp = self._upload(points, transform)
        if self.outlier_filter is not None:
            pts_wp = self.outlier_filter.apply(pts_wp)

        layers = self.builder.build(pts_wp)
        primary_layer = layers[self.primary]  # wp.array

        # Force a flat ground patch under the robot, in-place on the primary
        # reduction. Done BEFORE the raw_primary snapshot so the footprint reads
        # as "measured" to the support mask, and before inpaint so the patch acts
        # as a known boundary.
        if self.footprint_stamp is not None:
            self.footprint_stamp.apply(primary_layer, footprint_plane)

        # Inpaint mutates the primary buffer, so the stages snapshot the ORIGINAL
        # NaN-bearing primary first (raw_primary) — the support mask needs it to
        # know which cells were really measured.
        raw_primary, elevation, costs, inflated = self._run_grid_stages(primary_layer)

        traversability_wp: wp.array | None = None
        costs_wp: dict[str, wp.array] | None = None
        if self.analyzer is not None:
            total = costs.total
            if self.inflator is not None:
                # Inflate obstacles, then temporally gate, then mask by support ratio.
                # Inpainted cells with enough real neighbors keep their cost; the rest
                # become NaN, and frames that spike in obstacle count are rejected.
                if self.temporal_gate.decide():
                    total = self.support_mask.apply(raw_primary, inflated)
                else:
                    total = self.support_mask.rejected_frame()
            if self.occlusion_mask is not None:
                # Drop inpainted cost in the line-of-sight shadow of obstacles, so
                # the planner can't route through occluded space behind walls.
                total = self.occlusion_mask.apply(raw_primary, elevation, total)
            traversability_wp = total
            costs_wp = {"slope": costs.slope, "step": costs.step, "roughness": costs.roughness}

        return {
            "layers": layers,
            "elevation": elevation,
            "traversability": traversability_wp,
            "costs": costs_wp,
        }

    def _download(self, dev: dict[str, object]) -> TerrainMap:
        """Copy the selected device layers into a numpy-backed `TerrainMap`."""
        sel = self._layers
        layers = dev["layers"]
        costs_wp = dev["costs"]
        sources: dict[str, wp.array | None] = {
            "max": layers.max,
            "mean": layers.mean,
            "min": layers.min,
            "count": layers.count,
            "elevation": dev["elevation"],
            "slope_cost": costs_wp["slope"] if costs_wp is not None else None,
            "step_cost": costs_wp["step"] if costs_wp is not None else None,
            "roughness_cost": costs_wp["roughness"] if costs_wp is not None else None,
            "traversability": dev["traversability"],
        }
        sources = {k: v for k, v in sources.items() if k in sel and v is not None}

        tm = TerrainMap(resolution=self.resolution, bounds=self.bounds)
        if not self.device.is_cuda:
            for name, arr in sources.items():
                setattr(tm, name, arr.numpy().copy())
            return tm

        # Queue every D2H copy into pinned buffers, then wait once (blocking).
        for name, arr in sources.items():
            host = self._host_layers.get(name)
            if host is None or host.dtype != arr.dtype or host.shape != arr.shape:
                host = wp.empty(arr.shape, dtype=arr.dtype, device="cpu", pinned=True)
                self._host_layers[name] = host
            wp.copy(host, arr)
        cuda_sync.wait(self.device)
        for name in sources:
            setattr(tm, name, self._host_layers[name].numpy().copy())
        return tm

    def _wrap_device(self, dev: dict[str, object]) -> TerrainMapGPU:
        """Wrap the selected device layers as borrowed handles (no copy)."""
        sel = self._layers
        costs = dev["costs"]
        trav = dev["traversability"]
        return TerrainMapGPU(
            resolution=self.resolution,
            bounds=self.bounds,
            elevation=dev["elevation"] if "elevation" in sel else None,
            traversability=trav if (trav is not None and "traversability" in sel) else None,
            slope_cost=costs["slope"] if (costs is not None and "slope_cost" in sel) else None,
            step_cost=costs["step"] if (costs is not None and "step_cost" in sel) else None,
            roughness_cost=costs["roughness"] if (costs is not None and "roughness_cost" in sel) else None,
        )
