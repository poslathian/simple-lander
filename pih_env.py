"""Package-in-Hole gym environment.

The lander starts above the left pad, flies to a hole in the right pad,
picks up a package whose geometry and mass are only partially known, and
returns to the left pad.

Physics constants match kto_lander.py exactly.  The world is wider
(900×600 px = 30×20 m) than the standard 600×400 gymnasium LunarLander
so there is room for two pads and a mountain obstacle between them.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Optional

import Box2D
from Box2D.b2 import (
    contactListener, edgeShape, fixtureDef, polygonShape, revoluteJointDef,
)
import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Physics constants — must match kto_lander.py
# ---------------------------------------------------------------------------
FPS   = 50
DT    = 1.0 / FPS
SCALE = 30.0
GRAVITY = 10.0

MAIN_ENGINE_POWER     = 13.0
SIDE_ENGINE_POWER     = 0.6
MAIN_ENGINE_Y_OFF     = 4.0    # pixels, applied-point offset below body centre
SIDE_ENGINE_AWAY      = 12.0   # pixels, lateral offset of side-engine impulse point
SIDE_ENGINE_HEIGHT    = 14.0   # pixels, moment-arm height for side torque

# Lander and leg geometry (pixel units; divide by SCALE for world units)
LANDER_POLY = [(-14,+17), (-17,0), (-17,-10), (+17,-10), (+17,0), (+14,+17)]
LEG_AWAY = 20          # pixels — lateral distance from body centre to leg joint
LEG_DOWN = 18          # pixels — vertical distance from body centre to leg joint
LEG_W, LEG_H = 2, 8   # pixels
LEG_SPRING_TORQUE = 40

# ---------------------------------------------------------------------------
# World dimensions (wider than standard LunarLander to fit the PIH task)
# ---------------------------------------------------------------------------
VIEWPORT_W = 900
VIEWPORT_H = 600
WORLD_W = VIEWPORT_W / SCALE   # 30.0 m
WORLD_H = VIEWPORT_H / SCALE   # 20.0 m

# ---------------------------------------------------------------------------
# PIH task constants
# ---------------------------------------------------------------------------
PIH_START_X  = WORLD_W / 4          # 7.5 m  — left pad centre
PIH_PICKUP_X = 3 * WORLD_W / 4      # 22.5 m — right pad (pickup) centre
PIH_PAD_Y    = WORLD_H / 5          # 4.0 m  — elevation of both pads

PIH_MOUNTAIN_X          = WORLD_W / 2   # 15.0 m — mountain peak x
PIH_MOUNTAIN_H          = 5.0           # m above PIH_PAD_Y
PIH_MOUNTAIN_BASE_HALF  = 4.5           # m half-width at base

# PIH_LEG_OFFSET is the KTO planner's assumed leg-to-body vertical distance.
# The actual offset at spring equilibrium is ~0.44 m; proximity detection
# uses a ±0.5 m tolerance around true_contact_lander_y to cover this gap.
PIH_LEG_OFFSET = LEG_DOWN / SCALE   # 0.6 m

PIH_TIMEOUT = 30.0   # s — must exceed longest feasible KTO plan (~22 s)

RAY_MAX = 12.0   # world units — max raycast distance

# Short names used in waypoint labels during rendering
_WPT_SHORT: dict[str, str] = {
    'start':         'start',
    'mountain_out':  'mt_out',
    'approach':      'appr',
    'contact':       'ct',
    'extraction':    'ex',
    'mountain_ret':  'mt_ret',
    'approach_land': 'appr_l',
    'landing':       'land',
}


# ---------------------------------------------------------------------------
# Termination reasons
# ---------------------------------------------------------------------------
class TerminationReason(enum.Enum):
    NONE                 = "none"
    SUCCESS              = "success"
    CRASH                = "crash"
    TIMEOUT              = "timeout"
    FLYAWAY              = "flyaway"
    EXTRACTION_COLLISION = "extraction_collision"


# ---------------------------------------------------------------------------
# PIHConfig — single source of truth for all task geometry
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class PIHConfig:
    """True and assumed parameters for the Package-in-Hole task.

    True parameters are only used inside PackageInHoleEnv (physics, proximity
    detection, collision detection).  The planner must only access assumed_*
    properties.  This hard split is what Bug 2 (old implementation) violated.
    """
    hole_depth:             float = 1.0
    hole_width:             float = 2.0   # ≥ 2*(LEG_AWAY/SCALE + package_gap) for leg contact
    package_height_true:    float = 1.5   # total package height, metres
    package_mass_true:      float = 2.0   # kg
    package_height_assumed: float = 0.5   # assumed protrusion above hole rim, metres
    package_mass_assumed:   float = 3.0   # kg — conservative upper bound
    package_gap:            float = 0.05  # clearance between package and hole wall, metres
    n_raycast_rays:         int   = 8
    use_raycast_obs:        bool  = True
    start_at_pad:           bool  = False  # spawn at left pad instead of high altitude

    # ── True geometry — used by the environment ──────────────────────────

    @property
    def package_top_y(self) -> float:
        """True package-top elevation (world coords)."""
        return PIH_PAD_Y + (self.package_height_true - self.hole_depth)

    @property
    def true_contact_lander_y(self) -> float:
        """True lander-pivot y when legs reach the package top.

        Used as the centre of the proximity-attachment window.  The window
        is ±0.5 m to cover the ~0.16 m discrepancy between PIH_LEG_OFFSET
        (0.6 m, KTO model) and actual spring-equilibrium leg depth (~0.44 m).
        """
        return self.package_top_y + PIH_LEG_OFFSET

    @property
    def package_half_width(self) -> float:
        return (self.hole_width - 2 * self.package_gap) / 2

    @property
    def hole_half_width(self) -> float:
        return self.hole_width / 2

    # ── Assumed geometry — used by the planner ───────────────────────────

    @property
    def assumed_package_top_y(self) -> float:
        """Package top as believed by the KTO planner."""
        return PIH_PAD_Y + self.package_height_assumed

    @property
    def assumed_contact_lander_y(self) -> float:
        """Planned contact height: assumed package top + leg offset."""
        return self.assumed_package_top_y + PIH_LEG_OFFSET

    @property
    def extraction_lander_y(self) -> float:
        """Planned extraction height: package clears the hole under assumed geometry.

        Derivation (0.2 m margin above PIH_PAD_Y):
            pkg_bottom = extraction_lander_y - PIH_LEG_OFFSET
                         - (hole_depth + package_height_assumed)
                       = PIH_PAD_Y + 0.2   ✓

        Also the gate for extraction-collision detection: the check only
        fires once the lander reaches this height so that tracking drift
        during the ascent does not cause false positives.
        """
        return PIH_PAD_Y + self.hole_depth + self.package_height_assumed + 0.2 + PIH_LEG_OFFSET

    def with_true_as_assumed(self) -> "PIHConfig":
        """Return a copy where assumed parameters equal true values (oracle config)."""
        return dataclasses.replace(
            self,
            package_height_assumed=self.package_height_true,
            package_mass_assumed=self.package_mass_true,
        )


# ---------------------------------------------------------------------------
# Contact listener — terrain crash + leg ground contact only
# ---------------------------------------------------------------------------
class _ContactListener(contactListener):
    """Detects terrain contact for crash and leg-ground detection.

    Package contact is intentionally ignored here; attachment uses
    proximity detection to avoid Box2D leg-geometry inaccuracies.
    """

    def __init__(self, env: PackageInHoleEnv):
        contactListener.__init__(self)
        self.env = env

    def _is_terrain(self, body) -> bool:
        return body is self.env._terrain

    def BeginContact(self, contact):
        ba, bb = contact.fixtureA.body, contact.fixtureB.body
        # Lander body hits terrain → crash
        if ba is self.env.lander or bb is self.env.lander:
            if self._is_terrain(bb if ba is self.env.lander else ba):
                self.env.game_over = True
        # Leg ground contact
        for leg in self.env.legs:
            if leg is ba or leg is bb:
                if self._is_terrain(bb if ba is leg else ba):
                    leg.ground_contact = True

    def EndContact(self, contact):
        ba, bb = contact.fixtureA.body, contact.fixtureB.body
        for leg in self.env.legs:
            if leg is ba or leg is bb:
                if self._is_terrain(bb if ba is leg else ba):
                    leg.ground_contact = False


# ---------------------------------------------------------------------------
# Ray-cast callback
# ---------------------------------------------------------------------------
class _RayCastCallback(Box2D.b2RayCastCallback):
    def __init__(self, exclude):
        Box2D.b2RayCastCallback.__init__(self)
        self.fraction = 1.0
        self._exclude = exclude

    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.body in self._exclude:
            return 1.0
        self.fraction = min(self.fraction, fraction)
        return fraction


# ---------------------------------------------------------------------------
# Terrain geometry helper
# ---------------------------------------------------------------------------

def _terrain_y_at(x: float, cfg: PIHConfig) -> float:
    """Return terrain surface y-coordinate at world x.

    Matches the edge geometry built in _build_terrain().  Used by the renderer
    to compute package-ground penetration depth.
    """
    hhw = cfg.hole_half_width
    hd  = cfg.hole_depth
    if x <= PIH_START_X + 1.5:
        return PIH_PAD_Y
    if x <= PIH_MOUNTAIN_X:
        t = (x - (PIH_START_X + 1.5)) / (PIH_MOUNTAIN_X - (PIH_START_X + 1.5))
        return PIH_PAD_Y + t * PIH_MOUNTAIN_H
    if x <= PIH_PICKUP_X - 1.5:
        t = (x - PIH_MOUNTAIN_X) / (PIH_PICKUP_X - 1.5 - PIH_MOUNTAIN_X)
        return PIH_PAD_Y + PIH_MOUNTAIN_H * (1.0 - t)
    if x <= PIH_PICKUP_X - hhw:
        return PIH_PAD_Y
    if x <= PIH_PICKUP_X + hhw:
        return PIH_PAD_Y - hd   # hole floor
    return PIH_PAD_Y


# ---------------------------------------------------------------------------
# PackageInHoleEnv
# ---------------------------------------------------------------------------
class PackageInHoleEnv(gym.Env):
    """Gym environment for the Package-in-Hole task.

    Observation (base 12 dims, + n_raycast_rays if use_raycast_obs):
        0  (pos.x - WORLD_W/2) / (WORLD_W/2)         — normalised x
        1  (pos.y - PIH_PAD_Y) / (WORLD_H/2)          — normalised y above pad
        2  vel.x * (WORLD_W/2) / FPS                  — normalised vx
        3  vel.y * (WORLD_H/2) / FPS                  — normalised vy
        4  lander.angle                                 — radians
        5  20 * lander.angularVelocity / FPS           — normalised omega
        6  left leg ground contact (0/1)
        7  right leg ground contact (0/1)
        8  elapsed_s                                    — seconds
        9  attached (0/1)
       10  (package_top_y - PIH_PAD_Y) / (WORLD_H/2)  — normalised true protrusion
       11  (PIH_PICKUP_X - WORLD_W/2) / (WORLD_W/2)    — normalised pickup x (constant)
       12+ raycast distances [0,1], if use_raycast_obs

    Action: continuous [main_throttle, side] ∈ [-1,1]^2  (gymnasium convention)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        config: Optional[PIHConfig] = None,
        render_mode: Optional[str] = None,
        gravity: float = -GRAVITY,
    ):
        super().__init__()
        self.cfg = config if config is not None else PIHConfig()
        self.render_mode = render_mode
        self.gravity = gravity

        base_dim = 12
        ray_dim  = self.cfg.n_raycast_rays if self.cfg.use_raycast_obs else 0
        obs_dim  = base_dim + ray_dim

        low  = np.full(obs_dim, -np.inf, dtype=np.float32)
        high = np.full(obs_dim,  np.inf, dtype=np.float32)
        if self.cfg.use_raycast_obs:
            low[base_dim:]  = 0.0
            high[base_dim:] = 1.0
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space      = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.screen = None
        self.clock  = None

        # Set in reset()
        self.world    = None
        self.lander   = None
        self.legs     = []
        self._terrain = None
        self._package = None
        self.game_over    = False
        self.elapsed_s    = 0.0
        self._attached    = False
        self._termination_reason = TerminationReason.NONE
        self._kto_path_xy    = None   # sampled path for trajectory overlay
        self._oracle_path_xy = None   # oracle-corrected path (set on replan)
        self._waypoints      = None   # PIHWaypoints, injected by controller
        self._ctrl_t         = 0.0    # elapsed time for original plan reference dot
        self._plan_ref       = None   # original Plan callable, injected by controller
        self._oracle_plan_ref = None  # oracle Plan callable, set on replan
        self._oracle_ctrl_t  = 0.0   # elapsed time within oracle plan (resets on replan)
        self._show_raycasts   = False  # opt-in: visualise raycast beams
        self._font            = None   # lazy pygame font
        self._collision_side: float | None = None  # +1/-1 = which hole wall was hit

    # ── World construction ───────────────────────────────────────────────

    def _build_terrain(self):
        cfg  = self.cfg
        py   = PIH_PAD_Y
        hhw  = cfg.hole_half_width
        hd   = cfg.hole_depth
        mx   = PIH_MOUNTAIN_X
        mh   = PIH_MOUNTAIN_H
        mb   = PIH_MOUNTAIN_BASE_HALF

        self._terrain = self.world.CreateStaticBody()
        edges = [
            # left pad
            ((0,                       py), (PIH_START_X + 1.5,    py)),
            # mountain outbound slope
            ((PIH_START_X + 1.5,       py), (mx,                   py + mh)),
            # mountain return slope
            ((mx,                      py + mh), (PIH_PICKUP_X - 1.5, py)),
            # right pad left of hole
            ((PIH_PICKUP_X - 1.5,      py), (PIH_PICKUP_X - hhw,  py)),
            # hole left wall
            ((PIH_PICKUP_X - hhw,      py), (PIH_PICKUP_X - hhw,  py - hd)),
            # hole floor
            ((PIH_PICKUP_X - hhw,      py - hd), (PIH_PICKUP_X + hhw, py - hd)),
            # hole right wall
            ((PIH_PICKUP_X + hhw,      py - hd), (PIH_PICKUP_X + hhw, py)),
            # right pad right of hole
            ((PIH_PICKUP_X + hhw,      py), (WORLD_W,              py)),
        ]
        for p1, p2 in edges:
            self._terrain.CreateEdgeFixture(
                vertices=[p1, p2], density=0, friction=0.1,
                categoryBits=0x0001, maskBits=0x0030,
            )
        self._terrain.color1 = (0, 0, 0)
        self._terrain.color2 = (0, 0, 0)

    def _build_package(self):
        cfg = self.cfg
        py  = PIH_PAD_Y
        phw = cfg.package_half_width
        pht = cfg.package_height_true
        hd  = cfg.hole_depth

        # Package centre: x at pickup, y centred in hole + protrusion stack
        pkg_cx = PIH_PICKUP_X
        pkg_cy = py - hd + pht / 2.0

        self._package = self.world.CreateStaticBody(position=(pkg_cx, pkg_cy))
        self._package.CreateFixture(fixtureDef(
            shape=polygonShape(box=(phw, pht / 2.0)),
            density=0, friction=0.1,
            categoryBits=0x0004, maskBits=0x0020,   # collides with legs only
        ))
        self._package.color1 = (200, 150,  80)
        self._package.color2 = (160, 100,  40)

    def _build_lander(self, ix: float, iy: float):
        self.lander = self.world.CreateDynamicBody(
            position=(ix, iy),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in LANDER_POLY]),
                density=5.0, friction=0.1,
                categoryBits=0x0010, maskBits=0x0001,   # collides with terrain only
                restitution=0.0,
            ),
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = ( 77,  77, 128)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(ix - i * LEG_AWAY / SCALE, iy),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0, restitution=0.0,
                    categoryBits=0x0020, maskBits=0x0005,   # terrain + package
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = ( 77,  77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander, bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True, enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            leg.joint.motorEnabled = False
            self.legs.append(leg)

    def _destroy(self):
        if self._terrain is None:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self._terrain)
        self._terrain = None
        if self._package is not None:
            self.world.DestroyBody(self._package)
            self._package = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []

    # ── Observation ──────────────────────────────────────────────────────

    def _cast_ray(self, origin, angle_world: float) -> float:
        """Cast one ray; return normalised distance [0, 1]."""
        exclude = {self.lander} | set(self.legs)
        cb = _RayCastCallback(exclude)
        end = (
            origin[0] + math.cos(angle_world) * RAY_MAX,
            origin[1] + math.sin(angle_world) * RAY_MAX,
        )
        self.world.RayCast(cb, origin, end)
        return cb.fraction   # already in [0, 1]

    def _raycast_readings(self) -> np.ndarray:
        n   = self.cfg.n_raycast_rays
        pos = self.lander.position
        origin = (float(pos.x), float(pos.y))
        readings = np.empty(n, dtype=np.float32)
        for k in range(n):
            angle = self.lander.angle + (2 * math.pi * k / n)
            readings[k] = self._cast_ray(origin, angle)
        return readings

    def _build_obs(self) -> np.ndarray:
        pos = self.lander.position
        vel = self.lander.linearVelocity
        base = np.array([
            (pos.x - WORLD_W / 2) / (WORLD_W / 2),
            (pos.y - PIH_PAD_Y)   / (WORLD_H / 2),
            vel.x * (WORLD_W / 2) / FPS,
            vel.y * (WORLD_H / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.elapsed_s,
            1.0 if self._attached else 0.0,
            (self.cfg.package_top_y - PIH_PAD_Y) / (WORLD_H / 2),
            (PIH_PICKUP_X - WORLD_W / 2) / (WORLD_W / 2),
        ], dtype=np.float32)
        if self.cfg.use_raycast_obs:
            return np.concatenate([base, self._raycast_readings()])
        return base

    # ── Gymnasium interface ──────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()

        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self._terrain = None
        self._package = None
        self.game_over           = False
        self.elapsed_s           = 0.0
        self._attached           = False
        self._termination_reason = TerminationReason.NONE
        self._collision_side     = None

        self._build_terrain()
        self._build_package()

        if self.cfg.start_at_pad:
            ix = float(self.np_random.uniform(PIH_START_X - 0.5, PIH_START_X + 0.5))
            iy = float(PIH_PAD_Y + PIH_LEG_OFFSET + self.np_random.uniform(0.1, 0.3))
            self._build_lander(ix, iy)
            self.lander.linearVelocity = (0.0, 0.0)
            self.lander.angularVelocity = 0.0
        else:
            ix = float(self.np_random.uniform(PIH_START_X - 1.0, PIH_START_X + 1.0))
            iy = float(np.clip(
                WORLD_H * 0.85 + self.np_random.normal(0, 0.3),
                WORLD_H * 0.60, WORLD_H - 0.5,
            ))
            self._build_lander(ix, iy)
            self.lander.linearVelocity = (
                float(self.np_random.normal(0, 0.3)),
                float(self.np_random.normal(0, 0.3)),
            )
            self.lander.angularVelocity = float(self.np_random.normal(0, 0.05))

        cl = _ContactListener(self)
        self.world.contactListener_keepref = cl
        self.world.contactListener = cl

        self.drawlist = [self.lander] + self.legs
        if self._package is not None:
            self.drawlist.append(self._package)
        self._kto_path_xy     = None
        self._oracle_path_xy  = None
        self._waypoints       = None
        self._ctrl_t          = 0.0
        self._plan_ref        = None
        self._oracle_plan_ref = None
        self._oracle_ctrl_t   = 0.0
        # _show_raycasts and _font are not reset — caller sets once, font is cached

        if self.render_mode == "human":
            self.render()
        return self._build_obs(), {}

    def step(self, action):
        assert self.lander is not None
        action = np.clip(action, -1, +1).astype(np.float32)
        cfg = self.cfg

        # ── Engine forces — exact gymnasium LunarLander convention (no dispersion) ──
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])

        # Main engine: fires when action[0] > 0; m_power ∈ [0.5, 1.0].
        # oy uses -tip[1] so the impulse point is below the CoM and the
        # reaction force is upward (matching gymnasium's sign convention).
        if action[0] > 0.0:
            m_power = float((np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5)
            ox = float(tip[0] * MAIN_ENGINE_Y_OFF / SCALE)
            oy = float(-tip[1] * MAIN_ENGINE_Y_OFF / SCALE)
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power,
                 -oy * MAIN_ENGINE_POWER * m_power),
                (float(self.lander.position[0] + ox),
                 float(self.lander.position[1] + oy)), True,
            )

        # Side engine: fires when |action[1]| > 0.5; s_power ∈ [0.5, 1.0].
        # oy uses -side[1] to match gymnasium's sign convention.
        # The x impulse-point offset uses hardcoded 17/SCALE (not SIDE_ENGINE_HEIGHT)
        # matching gymnasium v1.2.3 exactly.
        if abs(action[1]) > 0.5:
            s_power = float(np.clip(abs(action[1]), 0.5, 1.0))
            direction = float(np.sign(action[1]))
            ox = float(side[0] * direction * SIDE_ENGINE_AWAY / SCALE)
            oy = float(-side[1] * direction * SIDE_ENGINE_AWAY / SCALE)
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power,
                 -oy * SIDE_ENGINE_POWER * s_power),
                (float(self.lander.position[0] + ox - tip[0] * 17 / SCALE),
                 float(self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)), True,
            )

        self.world.Step(DT, 6 * 30, 2 * 30)
        self.world.ClearForces()

        # Leg motor/density (same logic as gymnasium LunarLander)
        any_leg = self.legs[0].ground_contact or self.legs[1].ground_contact
        for leg in self.legs:
            leg.joint.motorEnabled = any_leg
            target_density = 1.0 if any_leg else 0.001
            f = leg.fixtures[0]
            if abs(f.density - target_density) > 0.01:
                f.density = target_density
                leg.ResetMassData()

        self.elapsed_s += DT
        pos = self.lander.position
        vel = self.lander.linearVelocity

        # ── Attachment: proximity-based ───────────────────────────────────
        # Use assumed_contact_lander_y (the height the planner targets) rather
        # than the true contact height.  The planner brings the lander to the
        # assumed position; the ±0.5 m window covers tracking error and the
        # 0.16 m leg-offset discrepancy.  Extraction-collision detection uses
        # the true package height separately, so sweep semantics are preserved.
        if not self._attached and self._package is not None:
            near_x    = abs(pos.x - PIH_PICKUP_X) < 1.0
            near_y    = abs(pos.y - cfg.assumed_contact_lander_y) < 0.5
            low_speed = abs(vel.y) < 1.5 and abs(vel.x) < 0.5
            if near_x and near_y and low_speed:
                self._attached = True
                md = self.lander.massData
                md.mass = md.mass + cfg.package_mass_true
                self.lander.massData = md
                self.world.DestroyBody(self._package)
                self._package = None
                self.drawlist = [self.lander] + self.legs

        # ── Extraction collision ──────────────────────────────────────────
        # Gate on extraction_lander_y (a plan event) not contact_lander_y.
        # Below extraction_lander_y the plan is still in its ascent phase;
        # any lateral drift there is tracking error, not a real wall strike.
        # Gate on x proximity to the hole: package-below-terrain elsewhere
        # (e.g. on the landing approach) is a landing geometry issue, not an
        # extraction failure — without this gate those episodes are mislabeled.
        if self._attached and pos.y >= cfg.extraction_lander_y \
                and abs(pos.x - PIH_PICKUP_X) < cfg.hole_width:
            pkg_bottom = pos.y - PIH_LEG_OFFSET - cfg.package_height_true
            pkg_in_hole = pkg_bottom < PIH_PAD_Y
            lateral_dev = abs(pos.x - PIH_PICKUP_X)
            if pkg_in_hole and lateral_dev > cfg.package_gap + 0.02:
                self._collision_side = 1.0 if pos.x > PIH_PICKUP_X else -1.0
                self._termination_reason = TerminationReason.EXTRACTION_COLLISION
                info = {"termination_reason": self._termination_reason}
                return self._build_obs(), -(PIH_TIMEOUT - self.elapsed_s) - DT, True, False, info

        obs = self._build_obs()

        # ── Termination ───────────────────────────────────────────────────
        near_ground = pos.y < PIH_PAD_Y + cfg.package_height_true + PIH_LEG_OFFSET + 0.05
        speed  = math.sqrt(vel.x ** 2 + vel.y ** 2)
        at_start = abs(pos.x - PIH_START_X) < 3.0
        landed = (
            self._attached and near_ground and not self.game_over
            and speed < 0.75 and abs(self.lander.angularVelocity) < 0.45
            and at_start
        )

        if landed:
            self._termination_reason = TerminationReason.SUCCESS
            terminated, reward = True, 0.0
        elif self.game_over or abs(obs[0]) >= 1.0:
            self._termination_reason = (
                TerminationReason.FLYAWAY if abs(obs[0]) >= 1.0
                else TerminationReason.CRASH
            )
            terminated = True
            reward = -(PIH_TIMEOUT - self.elapsed_s) - DT
        elif self.elapsed_s >= PIH_TIMEOUT:
            self._termination_reason = TerminationReason.TIMEOUT
            terminated = True
            reward = -(PIH_TIMEOUT - self.elapsed_s) - DT
        else:
            terminated = False
            reward = -DT

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, {"termination_reason": self._termination_reason}

    # ── Rendering ─────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode is None:
            return
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise gym.error.DependencyNotInstalled("pygame is not installed") from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self._font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._font = pygame.font.Font(None, 16)   # built-in, no fontconfig dep

        surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        # ── Black sky background (matches gymnasium LunarLander style) ────
        pygame.draw.rect(surf, (0, 0, 0), surf.get_rect())

        cfg = self.cfg
        py  = PIH_PAD_Y
        mx  = PIH_MOUNTAIN_X
        mh  = PIH_MOUNTAIN_H
        hhw = cfg.hole_half_width
        hd  = cfg.hole_depth

        def to_px(wx, wy):
            return (int(round(wx * SCALE)), int(round(VIEWPORT_H - wy * SCALE)))

        # ── White terrain polygon ─────────────────────────────────────────
        # Traces the surface profile left → right (including hole walls and
        # floor), then closes along the bottom of the viewport.  Filling
        # white on a black background mirrors the gymnasium sky-poly trick.
        terrain_poly = [
            to_px(0,                   py),
            to_px(PIH_START_X + 1.5,  py),
            to_px(mx,                  py + mh),
            to_px(PIH_PICKUP_X - 1.5, py),
            to_px(PIH_PICKUP_X - hhw, py),
            to_px(PIH_PICKUP_X - hhw, py - hd),
            to_px(PIH_PICKUP_X + hhw, py - hd),
            to_px(PIH_PICKUP_X + hhw, py),
            to_px(WORLD_W,             py),
            to_px(WORLD_W,             0.0),
            to_px(0.0,                 0.0),
        ]
        pygame.draw.polygon(surf, (255, 255, 255), terrain_poly)
        gfxdraw.aapolygon(surf, terrain_poly, (255, 255, 255))

        # ── Hole void: overdraw hole interior with black ──────────────────
        # The terrain polygon fills the hole cavity white; this restores it
        # to black so the hole reads as an opening in the pad.
        hole_void = [
            to_px(PIH_PICKUP_X - hhw, py),
            to_px(PIH_PICKUP_X - hhw, py - hd),
            to_px(PIH_PICKUP_X + hhw, py - hd),
            to_px(PIH_PICKUP_X + hhw, py),
        ]
        pygame.draw.polygon(surf, (0, 0, 0), hole_void)
        gfxdraw.aapolygon(surf, hole_void, (0, 0, 0))

        # ── Flags: two per pad, clamped to flat terrain ───────────────────
        # The mountain starts at PIH_START_X+1.5 and ends at PIH_PICKUP_X-1.5,
        # matching the hardcoded pad edges in _build_terrain().  Inner flags
        # are clamped to these boundaries so they never land on a slope.
        _FLAG_HALF   = 4 * LEG_AWAY / SCALE   # ≈ 2.67 m desired half-width
        _MT_LEFT     = PIH_START_X  + 1.5     # 9.0 m — left pad / mountain boundary
        _MT_RIGHT    = PIH_PICKUP_X - 1.5     # 21.0 m — mountain / right pad boundary
        for pad_cx in (PIH_START_X, PIH_PICKUP_X):
            fx_l = pad_cx - _FLAG_HALF
            fx_r = pad_cx + _FLAG_HALF
            if pad_cx == PIH_START_X:
                fx_r = min(fx_r, _MT_LEFT)    # clamp right flag to pad edge
            else:
                fx_l = max(fx_l, _MT_RIGHT)   # clamp left flag to pad edge
            for flag_wx in (fx_l, fx_r):
                sx, sy = to_px(flag_wx, PIH_PAD_Y)
                pygame.draw.line(surf, (255, 255, 255), (sx, sy), (sx, sy - 50), 1)
                tri = [(sx, sy - 50), (sx, sy - 40), (sx + 25, sy - 45)]
                pygame.draw.polygon(surf, (204, 204, 0), tri)
                gfxdraw.aapolygon(surf, tri, (204, 204, 0))

        # ── KTO trajectory line (original plan, cyan) ────────────────────
        if self._kto_path_xy is not None and len(self._kto_path_xy) > 1:
            pts = [to_px(*p) for p in self._kto_path_xy]
            pygame.draw.aalines(surf, (0, 180, 180), False, pts)

        # ── Oracle trajectory line (post-replan, orange) ─────────────────
        if self._oracle_path_xy is not None and len(self._oracle_path_xy) > 1:
            pts = [to_px(*p) for p in self._oracle_path_xy]
            pygame.draw.aalines(surf, (255, 140, 0), False, pts)

        # ── Waypoint circles + labels ─────────────────────────────────────
        if self._waypoints is not None:
            for f in dataclasses.fields(self._waypoints):
                wx, wy, _ws = getattr(self._waypoints, f.name)
                sx, sy = to_px(wx, wy)
                pygame.draw.circle(surf, (255, 200, 0), (sx, sy), 6)
                pygame.draw.circle(surf, (255, 255, 255), (sx, sy), 2)
                label = self._font.render(
                    _WPT_SHORT.get(f.name, f.name), True, (220, 220, 220)
                )
                surf.blit(label, (sx + 8, sy - 6))

        # ── Original plan reference dot (cyan — "where KTO thought") ────────
        if self._plan_ref is not None:
            ref = self._plan_ref(self._ctrl_t)
            sx, sy = to_px(float(ref[0]), float(ref[1]))
            pygame.draw.circle(surf, (0, 200, 200), (sx, sy), 5)

        # ── Oracle plan reference dot (green — "where oracle expects") ───────
        if self._oracle_plan_ref is not None:
            ref = self._oracle_plan_ref(self._oracle_ctrl_t)
            sx, sy = to_px(float(ref[0]), float(ref[1]))
            pygame.draw.circle(surf, (0, 220, 80), (sx, sy), 5)

        # ── Drawlist: lander, legs, package body (while not attached) ─────
        for obj in self.drawlist:
            for fix in obj.fixtures:
                if not hasattr(fix.shape, 'vertices'):
                    continue
                trans = fix.body.transform
                path  = [to_px(*trans * v) for v in fix.shape.vertices]
                pygame.draw.polygon(surf, obj.color1, path)
                gfxdraw.aapolygon(surf, path, obj.color2)

        # ── Package drawn at estimated lander-leg position when attached ──
        if self._attached and self.lander is not None:
            pos = self.lander.position
            pcx = float(pos.x)
            pcy = float(pos.y) - PIH_LEG_OFFSET - cfg.package_height_true / 2.0
            phw = cfg.package_half_width
            phh = cfg.package_height_true / 2.0
            corners = [
                to_px(pcx - phw, pcy + phh),
                to_px(pcx + phw, pcy + phh),
                to_px(pcx + phw, pcy - phh),
                to_px(pcx - phw, pcy - phh),
            ]
            pygame.draw.polygon(surf, (200, 150, 80), corners)
            gfxdraw.aapolygon(surf, corners, (160, 100, 40))

            # ── Package-ground penetration highlight ──────────────────────
            pkg_bottom = float(pos.y) - PIH_LEG_OFFSET - cfg.package_height_true
            terrain_y  = _terrain_y_at(pcx, cfg)
            if pkg_bottom < terrain_y:
                sliver = [
                    to_px(pcx - phw, pkg_bottom),
                    to_px(pcx + phw, pkg_bottom),
                    to_px(pcx + phw, terrain_y),
                    to_px(pcx - phw, terrain_y),
                ]
                pygame.draw.polygon(surf, (220, 50, 50), sliver)

        # ── Extraction collision marker ───────────────────────────────────
        if self._collision_side is not None and self.lander is not None:
            lpos = self.lander.position
            pkg_edge_x      = float(lpos.x) + self._collision_side * cfg.package_half_width
            pkg_bottom_now  = float(lpos.y) - PIH_LEG_OFFSET - cfg.package_height_true
            pkg_top_in_hole = min(PIH_PAD_Y, float(lpos.y) - PIH_LEG_OFFSET)
            contact_y       = (pkg_bottom_now + pkg_top_in_hole) / 2.0
            sx, sy = to_px(pkg_edge_x, contact_y)
            pygame.draw.circle(surf, (255, 50, 50), (sx, sy), 10)
            pygame.draw.circle(surf, (255, 200, 200), (sx, sy), 5)

        # ── Raycasts (opt-in: set env._show_raycasts = True) ─────────────
        if self._show_raycasts and self.lander is not None:
            n      = cfg.n_raycast_rays
            pos    = self.lander.position
            origin = (float(pos.x), float(pos.y))
            for k in range(n):
                angle = self.lander.angle + (2.0 * math.pi * k / n)
                frac  = self._cast_ray(origin, angle)
                ex    = origin[0] + math.cos(angle) * RAY_MAX * frac
                ey    = origin[1] + math.sin(angle) * RAY_MAX * frac
                color = (220, 60, 60) if frac < 0.5 else (160, 160, 160)
                pygame.draw.line(surf, color, to_px(*origin), to_px(ex, ey), 1)

        if self.render_mode == "human":
            self.screen.blit(surf, (0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit
            self.clock.tick(FPS)
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
