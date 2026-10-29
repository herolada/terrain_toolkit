"""Host ↔ GPU waits that sleep instead of spin.

With CUDA's default scheduling policy a host thread waiting on the GPU (context
or stream synchronize, synchronous device-to-host copy) busy-spins on its core.
For a small, launch-bound pipeline like this one that wait is a large part of
every frame, so it shows up as CPU load. Waiting on an event created with
CU_EVENT_BLOCKING_SYNC instead parks the thread in the OS until the GPU is done.
"""

from __future__ import annotations

import warp as wp

_CU_EVENT_BLOCKING_SYNC = 0x1
_CU_EVENT_DISABLE_TIMING = 0x2

_events: dict[str, wp.Event] = {}


def _blocking_event(device: wp.context.Device) -> wp.Event:
    ev = _events.get(device.alias)
    if ev is not None:
        return ev
    try:
        # wp.Event does not expose the blocking-sync flag, so create the raw
        # CUDA event through Warp's runtime and wrap it (lives for the process).
        from warp._src.context import runtime

        ptr = runtime.core.wp_cuda_event_create(
            device.context, _CU_EVENT_BLOCKING_SYNC | _CU_EVENT_DISABLE_TIMING,
        )
        ev = wp.Event(device, cuda_event=ptr) if ptr else wp.Event(device)
    except Exception:
        # Internal API moved: fall back to a regular (spinning) event.
        ev = wp.Event(device)
    _events[device.alias] = ev
    return ev


def wait(device: wp.context.Device | str | None = None) -> None:
    """Block (without spinning) until all work queued on `device`'s current
    stream has finished. No-op on CPU devices."""
    device = wp.get_device(device)
    if not device.is_cuda:
        return
    ev = _blocking_event(device)
    device.stream.record_event(ev)
    wp.synchronize_event(ev)
