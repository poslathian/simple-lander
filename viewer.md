# Live viewer — pushing your lander frames to the shared sprite

This branch (`simple-viewer`) adds a tiny frame hook to `lunar_lander.py` so
external code can grab the post-render `pygame.Surface` on every frame and
ship it somewhere. There is a hosted MJPEG viewer at:

```
https://lander-viewer-bnmbj.sprites.app/
```

If you push frames to its WebSocket endpoint, they show up live on that page.
This document explains exactly how to do that from your own script with the
**smallest possible diff** to your codebase: zero edits inside this repo
(the hook is already here on this branch), and one snippet you copy into
your driver script.

## Single-publisher caveat (read first)

> The sprite holds **one** "latest frame" slot. If two scripts publish at
> the same time, they'll fight over the slot — the viewer page will
> interleave frames from both publishers and look incoherent. The WS
> endpoint has no auth and no leasing right now. **Coordinate out of band**
> before you start streaming if more than one person uses this sprite.

## What this branch actually changes

`lunar_lander.py` gains a module-level frame hook (~17 lines, all in the
`simple-viewer` commit). With no hook installed (the default), behavior is
byte-identical to upstream `position-dagger`:

```python
# in lunar_lander.py
_FRAME_HOOK = None

def set_frame_hook(fn):
    """Register a callable receiving the post-render pygame.Surface, or None to clear."""
    global _FRAME_HOOK
    _FRAME_HOOK = fn
```

and one line at the end of `render()` after the existing `pygame.transform.flip`:

```python
        if _FRAME_HOOK is not None:
            _FRAME_HOOK(self.surf)
```

That's the whole footprint. Headless training / collection jobs that don't
render or don't call `set_frame_hook` are completely unaffected. The hook is
called from whichever thread is running `env.render()` — if your sim is
single-threaded, your hook is too.

## Checkout

```bash
git fetch origin simple-viewer
git checkout simple-viewer    # or merge/rebase the commit into your branch
```

If you'd rather cherry-pick just the hook into a different working branch:

```bash
git fetch origin simple-viewer
git cherry-pick origin/simple-viewer
```

The whole change is a single small commit (`Fix nothing; add 17 lines`).

## How to publish frames from your script

You need to do two things in your driver script:

1. **Install a hook** that snapshots the latest pygame surface into a
   thread-safe slot.
2. **Run a wall-clock timer** that wakes at your target rate, samples that
   slot, encodes the frame, and ships it over a WebSocket.

The "wall clock decoupled from sim rate" part matters: anything you call
between `env.step()`s blocks the sim thread, and several places in this
codebase block for 1–3 seconds at a time (e.g. `KTODiffusionController.warm_start`,
which solves a KTO program). If you encode-and-send synchronously inside
the hook, your wire rate collapses to the average sim throughput and the
viewer page goes dark whenever the sim is "thinking". Decoupling fixes this:
the hook does almost nothing, and a separate timer publishes the most recent
snapshot at a steady rate even when no new snapshots are arriving.

The minimal helper below is ~80 lines including comments. Save it as
`viewer_publisher.py` next to your script, or paste it inline if you prefer.

### `viewer_publisher.py`

```python
"""Minimal live-viewer publisher for the simple-viewer sprite.

Usage:
    import lunar_lander  # noqa: F401  (just to make sure it's importable)
    from viewer_publisher import start_publisher

    start_publisher(
        "wss://lander-viewer-bnmbj.sprites.app/ws",
        target_fps=10,     # wire rate; sim runs as fast as it likes
        downscale=2,       # 900x600 -> 450x300
    )

    # ...now run the env exactly as you normally do.
    # Every env.render() will snapshot the surface; the timer ships
    # whatever's latest at target_fps regardless of sim rate.
"""

from __future__ import annotations

import asyncio
import threading
import time
import zlib
from typing import Optional

import numpy as np
import pygame

import lunar_lander

# Wire format the sprite expects:
#   "LV1" (3 bytes) | width:u16 LE | height:u16 LE | seq:u32 LE | zlib(rgb24 row-major)
WIRE_MAGIC = b"LV1"


class _LatestSlot:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None

    def put(self, arr: np.ndarray) -> None:
        with self._lock:
            self._frame = arr

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame


def _encode(arr: np.ndarray, seq: int) -> bytes:
    h, w = arr.shape[:2]
    body = zlib.compress(arr.tobytes(), level=1)
    hdr = (
        WIRE_MAGIC
        + w.to_bytes(2, "little")
        + h.to_bytes(2, "little")
        + seq.to_bytes(4, "little")
    )
    return hdr + body


def start_publisher(url: str, *, target_fps: float = 10.0, downscale: int = 2) -> None:
    """Install the lander frame hook and start a background WS publisher.

    Call this once per process, BEFORE the first env.render().
    The publisher runs in a daemon thread; nothing to await, nothing to clean
    up. If the WS connection drops, the publisher reconnects with backoff.
    """
    slot = _LatestSlot()

    def _hook(surface: pygame.Surface) -> None:
        # surfarray.pixels3d returns (W, H, 3) — transpose to (H, W, 3)
        arr = np.transpose(np.array(pygame.surfarray.pixels3d(surface)), (1, 0, 2))
        if downscale > 1:
            arr = arr[::downscale, ::downscale, :].copy()
        slot.put(arr)

    lunar_lander.set_frame_hook(_hook)

    async def _ship() -> None:
        import websockets  # imported lazily so the dep is only required when streaming

        period = 1.0 / max(target_fps, 0.1)
        seq = 0
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
                    print(f"[viewer_publisher] connected to {url}", flush=True)
                    backoff = 1.0
                    next_tick = time.time()
                    while True:
                        next_tick += period
                        delay = next_tick - time.time()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        else:
                            next_tick = time.time()
                        arr = slot.get()
                        if arr is None:
                            continue
                        seq += 1
                        await ws.send(_encode(arr, seq))
            except Exception as exc:
                print(
                    f"[viewer_publisher] ws error: {exc!r}; reconnecting in {backoff:.1f}s",
                    flush=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 15.0)

    threading.Thread(
        target=lambda: asyncio.run(_ship()),
        name="lander-viewer-publisher",
        daemon=True,
    ).start()
```

### Dependencies

```bash
pip install websockets numpy
```

`pygame` is already pulled in by `gymnasium[box2d]` so you almost certainly
have it. `websockets` is the only thing most people will need to add.

### Putting it together

Whatever your driver script normally looks like — keyboard play, eval loop,
DAgger collector — drop the import + `start_publisher(...)` call near the
top, and don't change anything else:

```python
import gymnasium as gym
import lunar_lander
from viewer_publisher import start_publisher

start_publisher("wss://lander-viewer-bnmbj.sprites.app/ws", target_fps=10, downscale=2)

gym.register(id="LL-mine", entry_point="lunar_lander:LunarLander", max_episode_steps=1000)
env = gym.make("LL-mine", render_mode="rgb_array", continuous=True)

obs, _ = env.reset(seed=12345)
done = False
while not done:
    action = my_controller(obs)
    obs, r, term, trunc, _ = env.step(action)
    env.render()                # <- this fires the hook; viewer sees a frame
    done = term or trunc
```

If you're driving the env in `render_mode="human"` (a real pygame window),
that also works — `surf` is populated identically in both modes.

## Verifying it works

1. Open https://lander-viewer-bnmbj.sprites.app/ in any browser. You'll see
   either a gray placeholder (no publisher connected) or live frames from
   whoever is currently publishing.
2. Run your script. Within ~1 second of the first `env.render()` call you
   should see the page caption flip to `live • 450x300` and the frames start
   moving.
3. If it stays gray and your script is running, check (a) that
   `start_publisher` was called before `env.render()`, (b) that
   `pip show websockets` shows it installed, (c) that your network allows
   outbound WSS to `*.sprites.app`, and (d) that no other publisher is
   already connected (see caveat at the top).

## Stopping cleanly

The publisher thread is a daemon; when your process exits it goes with it.
There is no graceful shutdown — that's intentional, the goal here is "zero
ceremony to add live observability". The sprite-side server reuses the same
slot so leaving and rejoining a few seconds later just resumes streaming.

## Tunables

| arg / env var | what it does | default |
|---|---|---|
| `target_fps` | wall-clock rate at which the timer samples and ships | `10.0` |
| `downscale` | integer subsample of the rendered frame (use 1, 2, or 3) | `2` |
| `url` | WebSocket endpoint of the sprite | required |

The browser-side stream rate is set on the **sprite**, not on the publisher,
so even if you crank `target_fps` up the page will only refresh at whatever
the sprite's `STREAM_FPS` env var is set to (currently `10`). Lowering
`target_fps` below the sprite's stream rate will make the page show stale
frames between samples; raising it above wastes upload bandwidth. ~10 fps
matches the sprite default and is a good starting point.
