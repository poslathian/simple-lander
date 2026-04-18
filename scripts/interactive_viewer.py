"""Interactive replay viewer for PIH episode correction (Phase 2.5).

Pre-renders all frames from a failure episode (by replaying stored actions),
then lets an operator watch and click to place a user waypoint.  After
placement the previous named waypoint is identified and a re-solve is
triggered (stubbed until pih_solver.waypoint_replan_pih_with_kto is added).

Controls
--------
  Space        play / pause
  Left/Right   step one frame backward / forward
  [  ]         half / double playback speed (0.25× – 8×)
  Left-click   place user waypoint at clicked world coordinate
  W            run waypoint re-solve on placed waypoint (requires Drake)
  A            accept correction — replay with corrected plan and save record
  R            reject — clear waypoint, try again
  S            skip episode without correction
  N            next episode in the file
  Q / Esc      quit

Usage
-----
  python scripts/interactive_viewer.py \\
      --jsonl results/phase2_5/failures.jsonl \\
      --episode 0 \\
      --out   results/phase2_5/user_corrections.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pih_env import (
    PackageInHoleEnv, PIHConfig, TerminationReason,
    SCALE, VIEWPORT_W, VIEWPORT_H,
)
from pih_solver import PIHWaypoints, _numpy_to_plan, _WP_THRESHOLDS

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def to_px(wx: float, wy: float) -> tuple[int, int]:
    return (int(round(wx * SCALE)), int(round(VIEWPORT_H - wy * SCALE)))


def to_world(px_x: int, px_y: int) -> tuple[float, float]:
    return (px_x / SCALE, (VIEWPORT_H - px_y) / SCALE)


# ---------------------------------------------------------------------------
# Episode loading and frame pre-rendering
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Episode:
    seed:            int
    cfg:             PIHConfig
    states:          list[list[float]]   # per-step [x, y, vx, vy, theta, omega]
    actions:         list[list[float]]   # per-step [main, side]
    timestamps:      list[float]
    kto_cps:         np.ndarray          # (N, 2)
    kto_knots:       np.ndarray          # (N+4,)
    kto_plan_dict:   dict                # raw waypoint dict {name: [x,y,s]}
    waypoint_states: dict[str, list[float]]
    termination_reason: str
    raw:             dict                # full JSONL record


def load_episode(path: str | Path, idx: int) -> Episode:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    rec = records[idx]

    cfg_d = rec["cfg"]
    cfg = PIHConfig(
        hole_depth=cfg_d["hole_depth"],
        package_height_true=cfg_d["package_height_true"],
        package_height_assumed=cfg_d["package_height_assumed"],
        package_mass_true=cfg_d["package_mass_true"],
        package_mass_assumed=cfg_d["package_mass_assumed"],
        start_at_pad=cfg_d.get("start_at_pad", True),
    )

    cps   = np.array(rec["kto_plan"]["cps"],   dtype=float)
    knots = np.array(rec["kto_plan"]["knots"], dtype=float)

    return Episode(
        seed=rec["seed"],
        cfg=cfg,
        states=rec["states"],
        actions=rec["actions"],
        timestamps=rec["timestamps"],
        kto_cps=cps,
        kto_knots=knots,
        kto_plan_dict=rec["kto_plan"]["waypoints"],
        waypoint_states=rec.get("waypoint_states", {}),
        termination_reason=rec["termination_reason"],
        raw=rec,
    )


def count_episodes(path: str | Path) -> int:
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def prerender_frames(ep: Episode) -> list[np.ndarray]:
    """Replay episode with stored actions; return list of RGB frames."""
    env = PackageInHoleEnv(config=ep.cfg, render_mode="rgb_array")
    env.reset(seed=ep.seed)
    uw = env.unwrapped

    # Inject KTO arc into renderer
    plan = _numpy_to_plan(ep.kto_cps, ep.kto_knots)
    T    = plan.T
    uw._kto_path_xy = [
        (float(plan(s * T)[0]), float(plan(s * T)[1]))
        for s in np.linspace(0.0, 1.0, 80)
    ]

    # Inject named waypoints for circle rendering
    wp_d = ep.kto_plan_dict
    try:
        uw._waypoints = PIHWaypoints(**{
            k: tuple(float(v) for v in vals)
            for k, vals in wp_d.items()
        })
    except Exception:
        pass  # non-critical

    frames = []
    for i, act in enumerate(ep.actions):
        frame = env.render()
        if frame is not None:
            frames.append(frame.copy())
        env.step(np.array(act, dtype=np.float32))

    # Final frame after last step
    frame = env.render()
    if frame is not None:
        frames.append(frame.copy())

    env.close()
    return frames


# ---------------------------------------------------------------------------
# Waypoint helpers
# ---------------------------------------------------------------------------

def last_passed_waypoint(ep: Episode, frame_idx: int) -> Optional[str]:
    """Return name of the last named waypoint passed before frame_idx."""
    if not ep.waypoint_states:
        return None

    wp_order = list(ep.waypoint_states.keys())  # already in crossing order

    # Find which waypoints the lander reached by frame_idx
    # using stored waypoint_states: a waypoint is "passed" if the lander's
    # trajectory up to frame_idx came within threshold of its recorded position.
    passed = []
    for name in wp_order:
        ws = ep.waypoint_states[name]
        wx, wy = ws[0], ws[1]
        thresh = _WP_THRESHOLDS.get(name, 2.0)
        for state in ep.states[:frame_idx + 1]:
            dist = math.hypot(state[0] - wx, state[1] - wy)
            if dist < thresh:
                passed.append(name)
                break

    return passed[-1] if passed else None


# ---------------------------------------------------------------------------
# Overlay drawing (on a pygame Surface)
# ---------------------------------------------------------------------------

def draw_overlays(
    surf,                              # pygame.Surface
    ep: Episode,
    user_wp: Optional[tuple[float, float]],
    prev_wp_name: Optional[str],
    corrected_path: Optional[list[tuple[float, float]]],
    font,
    frame_idx: int,
    n_frames: int,
    playing: bool,
    speed: float,
    speed_flash: bool,
    status_msg: str,
) -> None:
    import pygame

    # ── User waypoint marker (yellow) ────────────────────────────────────
    if user_wp is not None:
        sx, sy = to_px(*user_wp)
        pygame.draw.circle(surf, (255, 220, 0), (sx, sy), 8, 2)
        pygame.draw.line(surf, (255, 220, 0), (sx - 12, sy), (sx + 12, sy), 1)
        pygame.draw.line(surf, (255, 220, 0), (sx, sy - 12), (sx, sy + 12), 1)

        # Line from previous waypoint to user waypoint
        if prev_wp_name and prev_wp_name in ep.waypoint_states:
            pws = ep.waypoint_states[prev_wp_name]
            px_p, py_p = to_px(pws[0], pws[1])
            pygame.draw.line(surf, (180, 180, 0), (px_p, py_p), (sx, sy), 1)

    # ── Corrected trajectory (magenta) ───────────────────────────────────
    if corrected_path and len(corrected_path) > 1:
        pts = [to_px(*p) for p in corrected_path]
        pygame.draw.aalines(surf, (220, 60, 220), False, pts)

    # ── HUD ─────────────────────────────────────────────────────────────
    t_sim = ep.timestamps[min(frame_idx, len(ep.timestamps) - 1)]
    speed_color = (255, 220, 60) if speed_flash else (200, 200, 200)
    hud_lines = [
        (f"Frame {frame_idx}/{n_frames - 1}  t={t_sim:.2f}s", (200, 200, 200)),
        (f"{'PLAYING' if playing else 'PAUSED'}  {speed:.2f}x", speed_color),
        (f"Seed {ep.seed}  {ep.termination_reason}", (200, 200, 200)),
    ]
    if user_wp:
        wx, wy = user_wp
        hud_lines.append((f"Waypoint ({wx:.2f}, {wy:.2f})", (200, 200, 200)))
        if prev_wp_name:
            hud_lines.append((f"Prev: {prev_wp_name}", (200, 200, 200)))
    if status_msg:
        hud_lines.append((status_msg, (200, 200, 200)))

    y_off = 6
    for line, color in hud_lines:
        txt = font.render(line, True, color)
        surf.blit(txt, (6, y_off))
        y_off += 14

    # ── Controls reminder ────────────────────────────────────────────────
    controls = "[Space] play/pause  [Left/Right] step  [[/]] speed (0.25x-8x)  [Click] waypoint"
    controls2 = "[W] re-solve  [A] accept  [R] reject  [S] skip  [N] next  [Q] quit"
    for i, c in enumerate([controls, controls2]):
        txt = font.render(c, True, (140, 140, 140))
        surf.blit(txt, (6, VIEWPORT_H - 24 + i * 12))


# ---------------------------------------------------------------------------
# Main viewer loop
# ---------------------------------------------------------------------------

def run_viewer(
    jsonl_path: str,
    start_episode: int,
    out_path: Optional[str],
) -> None:
    import pygame

    n_total = count_episodes(jsonl_path)
    ep_idx  = start_episode

    pygame.init()
    screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
    pygame.display.set_caption("PIH Correction Viewer")
    clock = pygame.time.Clock()
    font  = pygame.font.Font(None, 14)

    def load_and_render(idx: int):
        pygame.display.set_caption(f"PIH Viewer — loading episode {idx}…")
        ep = load_episode(jsonl_path, idx)
        frames = prerender_frames(ep)
        print(f"[viewer] ep {idx}  seed={ep.seed}  "
              f"{len(frames)} frames  {ep.termination_reason}")
        return ep, frames

    ep, frames = load_and_render(ep_idx)

    frame_idx   = 0
    playing     = False
    speed       = 1.0        # display FPS multiplier (base = 10 fps)
    BASE_FPS    = 10
    play_accum  = 0.0        # accumulated time for frame advance
    speed_flash_until = 0.0  # pygame.time.get_ticks() ms deadline for speed highlight

    user_wp:            Optional[tuple[float, float]] = None
    prev_wp_name:       Optional[str] = None
    corrected_path:     Optional[list[tuple[float, float]]] = None
    corrected_result:   Optional[dict] = None   # full waypoint_replan result dict
    frame_at_click:     int = 0
    status_msg = ""

    running = True
    while running:
        dt_ms = clock.tick(60)   # cap at 60 fps, get ms elapsed
        dt_s  = dt_ms / 1000.0

        # ── Events ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_SPACE:
                    playing = not playing
                    play_accum = 0.0

                elif event.key == pygame.K_RIGHT:
                    playing = False
                    frame_idx = min(frame_idx + 1, len(frames) - 1)

                elif event.key == pygame.K_LEFT:
                    playing = False
                    frame_idx = max(frame_idx - 1, 0)

                elif event.key == pygame.K_LEFTBRACKET:
                    speed = max(speed / 2.0, 0.25)
                    speed_flash_until = pygame.time.get_ticks() + 400

                elif event.key == pygame.K_RIGHTBRACKET:
                    speed = min(speed * 2.0, 8.0)
                    speed_flash_until = pygame.time.get_ticks() + 400

                elif event.key == pygame.K_s:
                    print(f"[viewer] skip ep {ep_idx}  seed={ep.seed}")
                    ep_idx = (ep_idx + 1) % n_total
                    ep, frames = load_and_render(ep_idx)
                    frame_idx = 0
                    user_wp = prev_wp_name = corrected_path = corrected_result = None
                    status_msg = ""

                elif event.key == pygame.K_n:
                    ep_idx = (ep_idx + 1) % n_total
                    ep, frames = load_and_render(ep_idx)
                    frame_idx = 0
                    user_wp = prev_wp_name = corrected_path = corrected_result = None
                    status_msg = ""

                elif event.key == pygame.K_r:
                    user_wp = prev_wp_name = corrected_path = corrected_result = None
                    status_msg = "Waypoint cleared."

                elif event.key == pygame.K_w:
                    if user_wp is None:
                        status_msg = "Place a waypoint first (left-click)."
                    else:
                        corrected_path, corrected_result, status_msg = _try_resolve(
                            ep, user_wp, prev_wp_name
                        )

                elif event.key == pygame.K_a:
                    if corrected_result is None:
                        status_msg = "Run re-solve first (W)."
                    else:
                        status_msg = _accept_correction(
                            ep, user_wp, prev_wp_name, frame_at_click,
                            corrected_result, out_path,
                        )
                        ep_idx = (ep_idx + 1) % n_total
                        ep, frames = load_and_render(ep_idx)
                        frame_idx = 0
                        user_wp = prev_wp_name = corrected_path = corrected_result = None

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                user_wp = to_world(mx, my)
                prev_wp_name = last_passed_waypoint(ep, frame_idx)
                frame_at_click = frame_idx
                corrected_path = corrected_result = None
                status_msg = (
                    f"Waypoint set. Prev: {prev_wp_name or 'none'}. Press W to re-solve."
                )

        # ── Playback advance ─────────────────────────────────────────────
        if playing:
            play_accum += dt_s * speed * BASE_FPS
            while play_accum >= 1.0 and frame_idx < len(frames) - 1:
                frame_idx += 1
                play_accum -= 1.0
            if frame_idx >= len(frames) - 1:
                playing = False

        # ── Render ──────────────────────────────────────────────────────
        base = frames[frame_idx]
        surf = pygame.surfarray.make_surface(base.swapaxes(0, 1))

        speed_flash = pygame.time.get_ticks() < speed_flash_until
        draw_overlays(
            surf, ep, user_wp, prev_wp_name, corrected_path,
            font, frame_idx, len(frames), playing, speed, speed_flash, status_msg,
        )

        screen.blit(surf, (0, 0))
        pygame.display.flip()

    pygame.quit()


# ---------------------------------------------------------------------------
# Re-solve (Component 2)
# ---------------------------------------------------------------------------

def _try_resolve(
    ep: Episode,
    user_wp: tuple[float, float],
    prev_wp_name: Optional[str],
) -> tuple[Optional[list[tuple[float, float]]], Optional[dict], str]:
    """Run waypoint_replan_pih_with_kto (requires Drake).

    Returns (corrected_path_for_display, full_result_dict, status_message).
    """
    if prev_wp_name is None:
        return None, None, "No previous waypoint — cannot re-solve."

    prev_state = ep.waypoint_states.get(prev_wp_name)
    if prev_state is None:
        return None, None, f"No recorded state for waypoint '{prev_wp_name}'."

    wp_order = ["mountain_out", "approach", "contact", "extraction", "mountain_ret", "landing"]
    try:
        prev_idx = wp_order.index(prev_wp_name)
        remaining = wp_order[prev_idx + 1:]
    except ValueError:
        remaining = []

    try:
        from pih_solver import waypoint_replan_pih_with_kto
        from kto_lander import LanderParams, compound_inertia_about_body_com
    except ImportError as exc:
        return None, None, f"Drake not available: {exc}"

    env = PackageInHoleEnv(config=ep.cfg, render_mode=None)
    env.reset(seed=ep.seed)
    mass, inertia = compound_inertia_about_body_com(env)
    params = LanderParams(mass=mass, inertia=inertia)
    env.close()

    result = waypoint_replan_pih_with_kto(
        user_waypoint=np.array(user_wp),
        previous_waypoint_name=prev_wp_name,
        previous_waypoint_state=np.array(prev_state),
        cfg=ep.cfg,
        remaining_waypoints=remaining,
        params=params,
        verbose=True,
    )

    if not result.get("feasible", False):
        return None, None, "Re-solve infeasible — try a different waypoint."

    plan   = result["plan"]
    plan_T = result["T"]
    path = [
        (float(plan(s * plan_T)[0]), float(plan(s * plan_T)[1]))
        for s in np.linspace(0.0, 1.0, 80)
    ]
    return path, result, f"Re-solve OK (T={plan_T:.1f}s). Press A to accept or R to reject."


# ---------------------------------------------------------------------------
# Accept (Components 3 + 4)
# ---------------------------------------------------------------------------

def _accept_correction(
    ep: Episode,
    user_wp: tuple[float, float],
    prev_wp_name: Optional[str],
    frame_at_click: int,
    corrected_result: dict,
    out_path: Optional[str],
) -> str:
    """Replay episode with corrected plan, then save the full record."""
    from replay_pih import replay_with_correction

    corrected_wpts_dict = corrected_result["waypoints"].to_dict()

    print(f"[viewer] replaying seed={ep.seed} with correction …")
    record = replay_with_correction(
        seed=ep.seed,
        cfg=ep.cfg,
        original_cps=ep.kto_cps,
        original_knots=ep.kto_knots,
        original_waypoints_dict=ep.kto_plan_dict,
        corrected_cps=corrected_result["cps"],
        corrected_knots=corrected_result["knots"],
        corrected_waypoints_dict=corrected_wpts_dict,
        previous_waypoint_name=prev_wp_name,
        user_waypoint=user_wp,
        frame_index=frame_at_click,
        verbose=True,
    )

    reason = record["termination_reason"]
    spliced = record["splice_happened"]

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[viewer] saved  seed={ep.seed}  reason={reason}  → {out_path}")

    outcome = "SUCCESS" if reason == "success" else reason
    return (f"Replay: {outcome}{'  (splice happened)' if spliced else '  (no splice!)'}. "
            f"Saved. Advancing.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl",    required=True, help="JSONL file of failure episodes")
    p.add_argument("--episode",  type=int, default=0, help="0-indexed episode to start at")
    p.add_argument("--out",      default=None, help="Output JSONL for saved corrections")
    args = p.parse_args()

    run_viewer(args.jsonl, args.episode, args.out)


if __name__ == "__main__":
    main()
