from __future__ import annotations

import math

import numpy as np
import warp as wp

from .kernels import (
    blur_axis_kernel,
    diffuse_step_kernel,
    downsample_kernel,
    finite_mask_kernel,
    upsample_inject_kernel,
)


def _as_wp_float32(heightmap: np.ndarray | wp.array) -> tuple[wp.array, bool]:
    """Return (wp.array view/copy, input_was_numpy)."""
    if isinstance(heightmap, wp.array):
        return heightmap, False
    if heightmap.ndim != 2:
        raise ValueError("heightmap must be 2D.")
    data = np.ascontiguousarray(heightmap, dtype=np.float32)
    return wp.array(data, dtype=wp.float32), True


def _diffuse(a: wp.array, b: wp.array, fixed: wp.array, iters: int) -> wp.array:
    """Run `iters` Jacobi steps ping-ponging between `a` and `b`; return the
    buffer holding the result."""
    dim = (a.shape[0], a.shape[1])
    for _ in range(iters // 2):
        wp.launch(diffuse_step_kernel, dim=dim, inputs=[a, fixed, b], device=a.device)
        wp.launch(diffuse_step_kernel, dim=dim, inputs=[b, fixed, a], device=a.device)
    if iters % 2:
        wp.launch(diffuse_step_kernel, dim=dim, inputs=[a, fixed, b], device=a.device)
        a, b = b, a
    return a


def _run_graphed(fn, device: wp.context.Device):
    """Run `fn()` once; on CUDA capture it into a graph first so its many small
    launches are replayed by the driver instead of issued one by one from Python."""
    if not device.is_cuda:
        return fn()
    with wp.ScopedCapture(device=device) as capture:
        result = fn()
    wp.capture_launch(capture.graph)
    return result


class MultigridInpainter:
    """Multigrid diffusion inpainting with a preallocated pyramid.

    `run(src)` fills NaN cells of `src` by solving the coarsest level, then
    upsampling and refining at each finer level. It mutates `src` (level 0 of the
    pyramid) and returns the buffer holding the result — either `src` or an
    internal scratch buffer. Only launches kernels (no allocation, no host sync),
    so a caller can capture it into a CUDA graph.
    """

    def __init__(
        self,
        height: int,
        width: int,
        *,
        iters_per_level: int = 50,
        coarse_iters: int = 200,
        min_size: int = 8,
        device: wp.context.Device | None = None,
    ):
        self.shape = (height, width)
        self.iters_per_level = iters_per_level
        self.coarse_iters = coarse_iters
        self.device = device if device is not None else wp.get_device()

        shapes = [self.shape]
        while shapes[-1][0] > min_size and shapes[-1][1] > min_size:
            h, w = shapes[-1]
            shapes.append(((h + 1) // 2, (w + 1) // 2))

        with wp.ScopedDevice(self.device):
            self._fixed = [wp.zeros(s, dtype=wp.int32) for s in shapes]
            self._tmp = [wp.zeros(s, dtype=wp.float32) for s in shapes]
            # Level 0 values are the caller's buffer; only coarse levels are owned.
            self._vals = [None] + [wp.zeros(s, dtype=wp.float32) for s in shapes[1:]]

    def run(self, src: wp.array) -> wp.array:
        if src.shape != self.shape:
            raise ValueError(f"heightmap shape {src.shape} != inpainter shape {self.shape}")
        dev = self.device
        vals = [src] + self._vals[1:]
        fixed = self._fixed
        wp.launch(finite_mask_kernel, dim=self.shape, inputs=[src, fixed[0]], device=dev)

        # Build pyramid (finest → coarsest).
        for lvl in range(1, len(vals)):
            wp.launch(downsample_kernel, dim=vals[lvl].shape,
                      inputs=[vals[lvl - 1], fixed[lvl - 1], vals[lvl], fixed[lvl]], device=dev)

        # Solve coarsest level, then upsample and refine from coarse to fine.
        top = len(vals) - 1
        solved = _diffuse(vals[top], self._tmp[top], fixed[top], self.coarse_iters)
        for lvl in range(top - 1, -1, -1):
            wp.launch(upsample_inject_kernel, dim=vals[lvl].shape,
                      inputs=[solved, vals[lvl], fixed[lvl]], device=dev)
            solved = _diffuse(vals[lvl], self._tmp[lvl], fixed[lvl], self.iters_per_level)
        return solved


def multigrid_inpaint(
    heightmap: np.ndarray | wp.array,
    iters_per_level: int = 50,
    coarse_iters: int = 200,
    min_size: int = 8,
) -> np.ndarray | wp.array:
    """Multigrid diffusion inpainting: solve coarse, upsample, refine at each level.

    Accepts numpy or `wp.array`; returns the matching type. Allocates a fresh
    pyramid each call — reuse a `MultigridInpainter` for repeated calls.
    """
    hm_wp, from_numpy = _as_wp_float32(heightmap)
    inpainter = MultigridInpainter(
        hm_wp.shape[0], hm_wp.shape[1],
        iters_per_level=iters_per_level, coarse_iters=coarse_iters,
        min_size=min_size, device=hm_wp.device,
    )
    out = _run_graphed(lambda: inpainter.run(hm_wp), hm_wp.device)
    if from_numpy:
        wp.synchronize()
        return out.numpy().copy()
    return out


def diffuse_inpaint(
    heightmap: np.ndarray | wp.array, max_iters: int = 500,
) -> np.ndarray | wp.array:
    """Fill NaN cells by iterative diffusion from known cells (Laplace inpainting).

    Uses CUDA graph capture on GPU to replay the iteration loop without per-launch
    Python overhead; on CPU falls back to plain `wp.launch` calls.
    """
    a, from_numpy = _as_wp_float32(heightmap)
    fixed = wp.zeros(a.shape, dtype=wp.int32, device=a.device)
    wp.launch(finite_mask_kernel, dim=a.shape, inputs=[a, fixed], device=a.device)
    b = wp.zeros(a.shape, dtype=wp.float32, device=a.device)
    a = _run_graphed(lambda: _diffuse(a, b, fixed, max_iters), a.device)
    if from_numpy:
        wp.synchronize()
        return a.numpy().copy()
    return a


class GaussianSmoother:
    """NaN-aware separable Gaussian blur (sigma in cells) with preallocated buffers.

    `run(src)` writes into an owned output buffer that is overwritten on the next
    call. Only launches kernels, so it can be captured into a CUDA graph.
    """

    def __init__(
        self,
        height: int,
        width: int,
        sigma: float = 1.0,
        truncate: float = 3.0,
        *,
        device: wp.context.Device | None = None,
    ):
        self.shape = (height, width)
        self.sigma = sigma
        self.device = device if device is not None else wp.get_device()
        self.radius = max(1, int(math.ceil(truncate * sigma))) if sigma > 0.0 else 0

        with wp.ScopedDevice(self.device):
            self._tmp = wp.zeros(self.shape, dtype=wp.float32)
            self._dst = wp.zeros(self.shape, dtype=wp.float32)
            self._weights = None
            if sigma > 0.0:
                k = np.arange(-self.radius, self.radius + 1, dtype=np.float32)
                weights = np.exp(-(k * k) / (2.0 * sigma * sigma)).astype(np.float32)
                self._weights = wp.array(weights, dtype=wp.float32)

    def run(self, src: wp.array) -> wp.array:
        if src.shape != self.shape:
            raise ValueError(f"heightmap shape {src.shape} != smoother shape {self.shape}")
        if self.sigma <= 0.0:
            # Zero-sigma fast path: copy so the caller can't alias the output.
            wp.copy(self._dst, src)
            return self._dst
        dev = self.device
        wp.launch(blur_axis_kernel, dim=self.shape,
                  inputs=[src, self._weights, self.radius, 1, self._tmp], device=dev)
        wp.launch(blur_axis_kernel, dim=self.shape,
                  inputs=[self._tmp, self._weights, self.radius, 0, self._dst], device=dev)
        return self._dst


def gaussian_smooth(
    heightmap: np.ndarray | wp.array, sigma: float = 1.0, truncate: float = 3.0,
) -> np.ndarray | wp.array:
    """NaN-aware separable Gaussian blur on a 2D heightmap (sigma in cells)."""
    if sigma <= 0.0 and not isinstance(heightmap, wp.array):
        return heightmap.copy()

    src, from_numpy = _as_wp_float32(heightmap)
    if len(src.shape) != 2:
        raise ValueError("heightmap must be 2D.")
    dst = GaussianSmoother(src.shape[0], src.shape[1], sigma, truncate, device=src.device).run(src)

    if from_numpy:
        wp.synchronize()
        return dst.numpy().copy()
    return dst
