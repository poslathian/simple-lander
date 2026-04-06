"""Simplified Lunar Lander — heuristic controller, time-optimal reward, no particles."""

import math
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle

try:
    import Box2D
    from Box2D.b2 import (
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e


FPS = 50
SCALE = 30.0
DT = 1.0 / FPS
TIMEOUT = 10.0

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 1.5

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = 4
GRAVITY = -10.0

VIEWPORT_W = 900
VIEWPORT_H = 600

# Derived Box2D constants (lander polygon density=5.0, two legs density=1.0)
SYSTEM_MASS = 4.958889
LANDER_BODY_MASS = 4.816667   # lander body only (impulses act on this mass)
LANDER_INERTIA_CM = 0.833315
LANDER_CM_LOCAL = (0.0, 0.101307)

# Leg spring damping: empirically fitted from Box2D motor + joint behaviour.
# With motors always enabled and full-mass legs, the joint motors + joint
# constraints act as a linear angular damper on the lander body:
#   alpha_spring ≈ -LEG_SPRING_DAMPING * omega   (R² ≈ 0.9998)
#
# The motor/joint coupling also reduces the effective angular gain from
# side thrust (~15x weaker than bare model, with sign reversal), but for
# the KTO solver we keep the bare torque_arm/I model since the PD tracking
# controller compensates for the mismatch and the damping dominates.
LEG_SPRING_DAMPING = 4.75      # rad/s² per rad/s (angular damping)

# Derived torque-arm constants for the side engine (Box2D impulse application point)
SIDE_ARM_A = 17.0 / SCALE - LANDER_CM_LOCAL[1]       # ≈ 0.465
SIDE_ARM_B = SIDE_ENGINE_HEIGHT / SCALE - LANDER_CM_LOCAL[1]  # ≈ 0.365
SIDE_AWAY_SCALED = SIDE_ENGINE_AWAY / SCALE           # = 0.400


def lander_dynamics(state, Fm, Fs):
    """Continuous-time dynamics of the lander (no collisions).

    State: [x, y, theta, vx, vy, omega]
    Controls: Fm (main thrust, >= 0), Fs (side thrust, signed)

    Returns: [dx/dt, dy/dt, dtheta/dt, dvx/dt, dvy/dt, domega/dt]

    Physics model (2D rigid-body rocket):
        Main engine force (body up):   Fm * (-sin θ,  cos θ)
        Side engine force (body right): Fs * ( cos θ, -sin θ)

        Torque from side engine + leg spring damping:
            α = Fs · torque_arm / I  -  LEG_SPRING_DAMPING · ω
    """
    x, y, theta, vx, vy, omega = state
    ct = math.cos(theta)
    st = math.sin(theta)

    m = LANDER_BODY_MASS
    g = abs(GRAVITY)
    I = LANDER_INERTIA_CM

    ax = (-Fm * st + Fs * ct) / m
    ay = (Fm * ct - Fs * st) / m - g

    torque_arm = (2 * SIDE_AWAY_SCALED * st * ct
                  + SIDE_ARM_A * st**2
                  - SIDE_ARM_B * ct**2)
    alpha = Fs * torque_arm / I - LEG_SPRING_DAMPING * omega

    return [vx, vy, omega, ax, ay, alpha]


def lander_acceleration(state, Fm, Fs):
    """Compute accelerations [ax, ay, alpha] from state and thrust.

    Same physics as lander_dynamics but returns only the acceleration vector,
    suitable for both continuous integration and discrete stepping.
    """
    x, y, theta, vx, vy, omega = state
    ct = math.cos(theta)
    st = math.sin(theta)

    m = LANDER_BODY_MASS
    g = abs(GRAVITY)
    I = LANDER_INERTIA_CM

    ax = (-Fm * st + Fs * ct) / m
    ay = (Fm * ct - Fs * st) / m - g

    torque_arm = (2 * SIDE_AWAY_SCALED * st * ct
                  + SIDE_ARM_A * st**2
                  - SIDE_ARM_B * ct**2)
    alpha = Fs * torque_arm / I - LEG_SPRING_DAMPING * omega

    return ax, ay, alpha


def lander_step(state, Fm, Fs, dt=DT):
    """One semi-implicit Euler step matching Box2D's integrator.

    State: [x, y, theta, vx, vy, omega]
    Returns: new state after dt.

    Box2D integration order:
      1. Compute acceleration from forces
      2. v_new = v + a * dt           (velocity update)
      3. x_new = x + v_new * dt       (position update with NEW velocity)

    This gives x_new = x + v*dt + a*dt², which is O(dt) more position change
    per step than exact integration (x + v*dt + 0.5*a*dt²).  Over many steps
    the difference compounds — this function matches Box2D exactly.
    """
    x, y, theta, vx, vy, omega = state
    ax, ay, alpha = lander_acceleration(state, Fm, Fs)

    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    omega_new = omega + alpha * dt

    x_new = x + vx_new * dt
    y_new = y + vy_new * dt
    theta_new = theta + omega_new * dt

    return [x_new, y_new, theta_new, vx_new, vy_new, omega_new]


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class LunarLander(gym.Env, EzPickle):
    """Simplified LunarLander with time-optimal reward.

    Observation (9-dim): x, y, vx, vy, angle, angular_vel, leg1, leg2, sim_t
    Reward: -dt per step. Crash/flyaway/timeout → total = -TIMEOUT. Landing → total ≈ -sim_t.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuous: bool = True,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        num_obstacles: int = 0,
    ):
        EzPickle.__init__(
            self, render_mode, continuous, gravity, enable_wind,
            wind_power, turbulence_power, num_obstacles,
        )
        assert -12.0 < gravity < 0.0
        self.gravity = gravity
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power
        assert 0 <= num_obstacles <= 5
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.enable_wind = enable_wind
        self.continuous = continuous

        self.screen = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.lander = None

        # 9-dim obs: x, y, vx, vy, angle, angular_vel, leg1, leg2, sim_t
        low = np.array([-2.5, -2.5, -10, -10, -2 * math.pi, -10, 0, 0, 0], dtype=np.float32)
        high = np.array([2.5, 2.5, 10, 10, 2 * math.pi, 10, 1, 1, TIMEOUT], dtype=np.float32)
        self.observation_space = spaces.Box(low, high)

        if self.continuous:
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)

        self.render_mode = render_mode
        self.m_power = 0.0
        self.s_power = 0.0
        self.s_dir = 0.0

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        for obs in self.obstacles:
            self.world.DestroyBody(obs)
        self.obstacles = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.elapsed_s = 0.0
        self.m_power = 0.0
        self.s_power = 0.0
        self.s_dir = 0.0

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.moon.color1 = (0, 0, 0)
        self.moon.color2 = (0, 0, 0)

        # Lander — uniform across full display width, near top
        initial_x = float(self.np_random.uniform(1.0, W - 1.0))
        initial_y = float(np.clip(H * 0.85 + self.np_random.normal(0, 0.3), H * 0.6, H - 0.5))

        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=float(self.np_random.normal(0, 0.05)),
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x003,
                restitution=0.0,
            ),
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)
        self.lander.linearVelocity = (
            float(self.np_random.normal(0, 0.3)),
            float(self.np_random.normal(0, 0.3)),
        )
        self.lander.angularVelocity = float(self.np_random.normal(0, 0.1))

        if self.enable_wind:
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)

        # Legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x003,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
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
            self.legs.append(leg)

        self._create_obstacles()
        self.drawlist = [self.lander] + self.legs + self.obstacles

        if self.render_mode == "human":
            self.render()
        return self._build_obs(), {}

    def _create_obstacles(self):
        """Create satellite obstacles as static Box2D bodies."""
        self.obstacles = []
        self.obstacle_radii = []
        if self.num_obstacles == 0:
            return

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        pad_cx = (self.helipad_x1 + self.helipad_x2) / 2
        BASE_BOUNDING_RADIUS = 0.75

        positions, scales = [], []
        for _ in range(self.num_obstacles):
            r = float(self.np_random.uniform(0.5, 2.0))
            for _attempt in range(50):
                x = self.np_random.uniform(1.0, W - 1.0)
                obs_y_lo = self.helipad_y + 2.0
                obs_y_hi = H - 2.0
                shift = 0.20 * (obs_y_hi - obs_y_lo)
                y = self.np_random.uniform(obs_y_lo + shift, obs_y_hi)
                if (x - pad_cx) ** 2 + (y - self.helipad_y) ** 2 < 3.0 ** 2:
                    continue
                sep = 2.0 * BASE_BOUNDING_RADIUS * r + 1.0
                too_close = any(
                    (x - px) ** 2 + (y - py) ** 2 < (sep + BASE_BOUNDING_RADIUS * sr) ** 2
                    for (px, py), sr in zip(positions, scales)
                )
                if too_close:
                    continue
                positions.append((x, y))
                scales.append(r)
                break

        obs_fix = dict(categoryBits=0x0002, maskBits=0x0030)
        for (ox, oy), r in zip(positions, scales):
            ob = self.world.CreateStaticBody(position=(ox, oy))
            ob.CreateFixture(fixtureDef(shape=polygonShape(box=(0.20 * r, 0.20 * r)), **obs_fix))
            ob.CreateFixture(fixtureDef(shape=polygonShape(box=(0.275 * r, 0.065 * r, (-0.475 * r, 0), 0)), **obs_fix))
            ob.CreateFixture(fixtureDef(shape=polygonShape(box=(0.275 * r, 0.065 * r, (0.475 * r, 0), 0)), **obs_fix))
            ob.CreateFixture(fixtureDef(shape=polygonShape(box=(0.05 * r, 0.14 * r, (0, 0.34 * r), 0)), **obs_fix))
            ob.color1 = (204, 204, 204)
            ob.color2 = (102, 102, 102)
            ob.bounding_radius = BASE_BOUNDING_RADIUS * r
            self.obstacles.append(ob)
            self.obstacle_radii.append(BASE_BOUNDING_RADIUS * r)

    def _build_obs(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity
        return np.array([
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.elapsed_s,
        ], dtype=np.float32)

    def step(self, action):
        assert self.lander is not None

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)

        # Wind
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + math.sin(math.pi * 0.01 * self.wind_idx)
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter((wind_mag, 0.0), True)
            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + math.sin(math.pi * 0.01 * self.torque_idx)
                )
                * self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(torque_mag, True)

        # Engine forces (no dispersion)
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])

        # Main engine — linear over [-1, 1]: action=-1 → off, action=1 → full
        if self.continuous:
            self.m_power = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        else:
            self.m_power = 1.0 if action == 2 else 0.0

        if self.m_power > 0.0:
            ox = tip[0] * MAIN_ENGINE_Y_LOCATION / SCALE
            oy = -tip[1] * MAIN_ENGINE_Y_LOCATION / SCALE
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            fx = float(-ox * MAIN_ENGINE_POWER * self.m_power)
            fy = float(-oy * MAIN_ENGINE_POWER * self.m_power)
            self.lander.ApplyLinearImpulse((fx, fy), impulse_pos, True)

        # Side engines — linear over [-1, 1]: sign = direction, magnitude = power
        if self.continuous:
            self.s_dir = float(np.sign(action[1])) if abs(float(action[1])) > 1e-6 else 0.0
            self.s_power = float(np.clip(np.abs(action[1]), 0.0, 1.0))
        else:
            if action in [1, 3]:
                self.s_dir = float(action - 2)
                self.s_power = 1.0
            else:
                self.s_dir = 0.0
                self.s_power = 0.0

        if self.s_power > 0.0:
            ox = side[0] * SIDE_ENGINE_AWAY / SCALE
            oy = -side[1] * SIDE_ENGINE_AWAY / SCALE
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            fx = float(-self.s_dir * self.s_power * SIDE_ENGINE_POWER * side[0])
            fy = float(-self.s_dir * self.s_power * SIDE_ENGINE_POWER * side[1])
            self.lander.ApplyLinearImpulse((fx, fy), impulse_pos, True)

        # Physics step
        self.world.Step(1.0 / FPS, 6, 2)
        self.world.ClearForces()

        # Leg joint motors stay enabled always (full-mass legs) — they provide
        # passive angular damping modelled as LEG_SPRING_DAMPING in the surrogate.

        self.elapsed_s += DT
        state = self._build_obs()

        # Reward: -dt per step; crash/flyaway/timeout total = -TIMEOUT; landing total ≈ -sim_t
        reward = -DT

        terminated = False
        both_legs = self.legs[0].ground_contact and self.legs[1].ground_contact
        vel = self.lander.linearVelocity
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2)
        landed = (
            not self.game_over
            and both_legs
            and speed < 0.5
            and abs(self.lander.angularVelocity) < 0.3
        )
        crashed = self.game_over or abs(state[0]) >= 1.0
        timed_out = self.elapsed_s >= TIMEOUT

        if landed or not self.lander.awake:
            terminated = True
        if crashed or timed_out:
            terminated = True

        if terminated and not landed:
            # Make total reward = -TIMEOUT
            reward += -(TIMEOUT - self.elapsed_s)

        if self.render_mode == "human":
            self.render()

        return state, reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        # Sky / terrain
        for p in self.sky_polys:
            scaled = [(c[0] * SCALE, c[1] * SCALE) for c in p]
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled)
            gfxdraw.aapolygon(self.surf, scaled, (0, 0, 0))

        # Bodies (lander, legs, obstacles)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v * SCALE for v in f.shape.vertices]
                pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                gfxdraw.aapolygon(self.surf, path, obj.color1)
                pygame.draw.aalines(self.surf, color=obj.color2, points=path, closed=True)

        # Helipad flags
        for x in [self.helipad_x1, self.helipad_x2]:
            x = x * SCALE
            flagy1 = self.helipad_y * SCALE
            flagy2 = flagy1 + 50
            pygame.draw.line(self.surf, (255, 255, 255), (x, flagy1), (x, flagy2), 1)
            pygame.draw.polygon(
                self.surf, (204, 204, 0),
                [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
            )

        # KTO planned trajectory (spline + knots)
        if hasattr(self, '_kto_path_xy') and self._kto_path_xy is not None:
            path = self._kto_path_xy
            knots = self._kto_knot_xy
            # Draw spline as a line (cyan)
            if len(path) > 1:
                pts = [(float(p[0] * SCALE), float(p[1] * SCALE)) for p in path]
                pygame.draw.aalines(self.surf, (0, 200, 200), False, pts)
            # Draw knot points as dots (magenta)
            for kx, ky in knots:
                sx, sy = int(kx * SCALE), int(ky * SCALE)
                if 0 <= sx < VIEWPORT_W and 0 <= sy < VIEWPORT_H:
                    gfxdraw.aacircle(self.surf, sx, sy, 3, (255, 0, 255))
                    gfxdraw.filled_circle(self.surf, sx, sy, 3, (255, 0, 255))

        # Thrust triangles (4 indicators: up, down, left, right)
        if self.lander is not None:
            self._draw_thrust(self.surf)

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.close()
                    raise SystemExit
            target_fps = self.metadata["render_fps"] * getattr(self, "speedup", 1.0)
            if target_fps > 0:
                self.clock.tick(target_fps)
            # speedup=0 → no tick limit (fast as possible)
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def _draw_thrust(self, surf):
        """Draw 4 thrust indicator triangles around the lander."""
        import pygame

        lx = self.lander.position[0] * SCALE
        ly = self.lander.position[1] * SCALE
        a = self.lander.angle
        ca, sa = math.cos(a), math.sin(a)

        def rot(dx, dy):
            return (float(lx + dx * ca - dy * sa), float(ly + dx * sa + dy * ca))

        ACTIVE = (255, 140, 0)
        DIM = (60, 60, 60)

        # Up thrust (main engine exhaust below lander)
        color = ACTIVE if self.m_power > 0.05 else DIM
        sz = 10 + 18 * self.m_power
        tri = [rot(-5, -14), rot(5, -14), rot(0, -14 - sz)]
        pygame.draw.polygon(surf, color, tri)

        # Down indicator (above lander, never active)
        tri = [rot(-5, 20), rot(5, 20), rot(0, 30)]
        pygame.draw.polygon(surf, DIM, tri)

        # Left indicator
        active_left = self.s_power > 0.05 and self.s_dir < 0
        color = ACTIVE if active_left else DIM
        sz = 5 + 12 * (self.s_power if active_left else 0)
        tri = [rot(-14, -4), rot(-14, 4), rot(-14 - sz, 0)]
        pygame.draw.polygon(surf, color, tri)

        # Right indicator
        active_right = self.s_power > 0.05 and self.s_dir > 0
        color = ACTIVE if active_right else DIM
        sz = 5 + 12 * (self.s_power if active_right else 0)
        tri = [rot(14, -4), rot(14, 4), rot(14 + sz, 0)]
        pygame.draw.polygon(surf, color, tri)

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


# ---------------------------------------------------------------------------
# Heuristic controller
# ---------------------------------------------------------------------------

def heuristic(env, s):
    """PD heuristic controller.

    Args:
        s: 9-dim observation [x, y, vx, vy, angle, ang_vel, leg1, leg2, sim_t]
    """
    angle_targ = s[0] * 0.5 + s[2] * 1.0
    angle_targ = np.clip(angle_targ, -0.4, 0.4)
    hover_targ = 0.55 * np.abs(s[0])

    # Time pressure: descend more aggressively as timeout approaches
    time_left = TIMEOUT - s[8]
    if time_left < 4.0:
        hover_targ *= max(0.2, time_left / 4.0)

    angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = (0 - s[4]) * 0.5 - s[5] * 1.0  # keep leveling
        hover_todo = -s[3] * 0.8  # moderate braking, let settle

    if env.unwrapped.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


# ---------------------------------------------------------------------------
# KTO tracking controller
# ---------------------------------------------------------------------------

class KTOController:
    """KTO trajectory controller: PD tracking + heuristic settle.

    Phase 1: Track the KTO plan with cascaded PD feedback control.
              Outer loop corrects position/velocity → desired acceleration.
              Inner loop corrects attitude → side thrust.
    Phase 2: Heuristic PD controller for final descent and landing.
    """

    def __init__(self, env, time_budget=5.0, warmstart_budget=1.0):
        import solver

        uw = env.unwrapped
        # Zero initial velocity to match solver boundary conditions
        uw.lander.linearVelocity = (0.0, 0.0)
        uw.lander.angularVelocity = 0.0

        pos = uw.lander.position
        start = np.array([pos.x, pos.y, uw.lander.angle])
        # Target close to the pad — heuristic handles final touchdown
        goal_y = uw.helipad_y + LEG_DOWN / SCALE + 1.0
        goal = np.array([solver.PAD_X, goal_y, 0.0])
        goal_vel = np.array([0.0, -0.5, 0.0])  # small downward velocity

        obstacle_tuples = []
        for obs_body, r in zip(uw.obstacles, getattr(uw, "obstacle_radii", [])):
            obstacle_tuples.append((obs_body.position.x, obs_body.position.y, r))

        times, plan, constraint_xy, warm_xy, knot_xy, control_xy, *_ = solver.solve(
            start=start, goal=goal,
            goal_velocity=goal_vel,
            obstacles=tuple(obstacle_tuples),
            time_budget=time_budget,
            warmstart_budget=warmstart_budget,
        )

        self.plan_times = times
        self.plan = plan
        self.gains = solver.DEFAULT_GAINS

        # Forces are DT-aligned: n_steps entries; positions have n_steps+1
        self.n_steps = len(plan["Fm"])
        self.idx = 0

        # Plan positions for rendering/tests (direct from DT-aligned sampling)
        self.plan_x = plan["x"]
        self.plan_y = plan["y"]
        self.plan_theta = plan["theta"]

        # Store trajectory for rendering: spline path (x,y) and knot points
        self.path_xy = np.column_stack([plan["x"], plan["y"]])
        self.knot_xy = knot_xy

        # Attach to env so render() can draw it
        uw._kto_path_xy = self.path_xy
        uw._kto_knot_xy = self.knot_xy

    def step(self, env):
        """PD tracking controller: compute corrective actions from plan."""
        import solver

        if self.idx < self.n_steps:
            i = self.idx
            self.idx += 1

            # Read actual state from Box2D
            uw = env.unwrapped
            L = uw.lander
            x, y = L.position.x, L.position.y
            theta = L.angle
            vx, vy = L.linearVelocity.x, L.linearVelocity.y
            omega = L.angularVelocity

            # Reference from DT-aligned plan (direct index)
            p = self.plan
            x_ref = float(p["x"][i])
            y_ref = float(p["y"][i])
            th_ref = float(p["theta"][i])
            vx_ref = float(p["vx"][i])
            vy_ref = float(p["vy"][i])
            om_ref = float(p["omega"][i])
            ax_ref = float(p["ax"][i])
            ay_ref = float(p["ay"][i])
            al_ref = float(p["alpha"][i])

            Fm, Fs = solver._tracking_step(
                x, y, theta, vx, vy, omega,
                x_ref, y_ref, th_ref, vx_ref, vy_ref, om_ref,
                ax_ref, ay_ref, al_ref, self.gains)

            # Convert to actions
            a_main = float(np.clip(2.0 * Fm / solver.THRUST_MAX - 1.0, -1.0, 1.0))
            a_side = float(np.clip(Fs / solver.SIDE_FORCE_MAX, -1.0, 1.0))

            return np.array([a_main, a_side], dtype=np.float32)
        else:
            return heuristic(env, env.unwrapped._build_obs())


# ---------------------------------------------------------------------------
# Main — heuristic by default, --keyboard for manual control
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simplified Lunar Lander")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=0, help="0 = infinite")
    parser.add_argument("--keyboard", action="store_true",
                        help="Manual control (arrow keys / WASD)")
    parser.add_argument("--kto", action="store_true",
                        help="KTO trajectory solver (open-loop + heuristic settle)")
    parser.add_argument("--obstacles", type=int, default=0,
                        help="Number of obstacles (0-5)")
    parser.add_argument("--save-frames", type=str, default=None,
                        help="Save frames to directory (e.g. ./tmp)")
    parser.add_argument("--speedup", type=float, default=1.0,
                        help="0=fast as possible, 1.0=realtime 50fps, 2.0=2x faster")
    args = parser.parse_args()

    render_mode = "human"
    if args.save_frames:
        render_mode = "rgb_array"

    gym.register(
        id="LunarLander-simple",
        entry_point="lunar_lander:LunarLander",
        max_episode_steps=1000,
        kwargs={"num_obstacles": args.obstacles},
    )
    env = gym.make("LunarLander-simple", render_mode=render_mode, continuous=True)
    env.unwrapped.speedup = args.speedup

    # Keyboard state
    _kb = {"action": np.zeros(2, dtype=np.float32), "quit": False}

    def poll_keyboard():
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _kb["quit"] = True
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                _kb["quit"] = True
        keys = pygame.key.get_pressed()
        main = 1.0 if (keys[pygame.K_UP] or keys[pygame.K_w]) else -1.0
        side = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            side = -1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            side = 1.0
        _kb["action"] = np.array([main, side], dtype=np.float32)

    if args.keyboard:
        print("KEYBOARD MODE: Up/W=thrust, Left-Right/A-D=rotate, Q/Esc=quit")

    if args.save_frames:
        import os
        os.makedirs(args.save_frames, exist_ok=True)

    episode = 0
    while args.episodes == 0 or episode < args.episodes:
        obs, _ = env.reset(seed=args.seed + episode)
        total_reward, done, steps = 0.0, False, 0
        frame_idx = 0

        kto_ctrl = None
        if args.kto:
            import time as _time
            t0 = _time.monotonic()
            kto_ctrl = KTOController(env, time_budget=5.0)
            print(f"  KTO solve: {_time.monotonic() - t0:.2f}s, "
                  f"{kto_ctrl.n_steps} steps planned")

        while not done:
            if args.keyboard:
                action = _kb["action"]
            elif kto_ctrl is not None:
                action = kto_ctrl.step(env)
            else:
                action = heuristic(env, obs)

            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = term or trunc

            if args.save_frames:
                img = env.render()
                if img is not None:
                    from PIL import Image
                    Image.fromarray(img).save(
                        os.path.join(args.save_frames, f"ep{episode:02d}_f{frame_idx:04d}.png")
                    )
                    frame_idx += 1

            if args.keyboard:
                poll_keyboard()
                if _kb["quit"]:
                    env.close()
                    exit()

        uw = env.unwrapped
        landed = (
            not uw.game_over
            and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
        )
        status = "LANDED" if landed else "CRASHED/TIMEOUT"
        print(f"Ep {episode}: {status}  steps={steps}  t={uw.elapsed_s:.2f}s  reward={total_reward:.2f}")
        episode += 1

    env.close()
