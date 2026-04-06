"""Tests for the DiffusionController pipeline with KTO guidance."""

import numpy as np
import gymnasium as gym

from lunar_lander import LunarLander, KTOController, heuristic, DT, TIMEOUT
from diffusion_controller import DiffusionController, LanderState, ActionTarget
from guidance_controller import GuidanceController


def obs_to_lander_state(obs):
    return LanderState(
        t_sim_lander=float(obs[8]),
        q=(float(obs[0]), float(obs[1]), float(obs[4])),
        q_prime=(float(obs[2]), float(obs[3]), float(obs[5])),
        thrust=(0.0, 0.0),
        contacts=(bool(obs[6]), bool(obs[7]), False),
    )


def rollout_kto(env, seed):
    """Run one episode with standalone KTO controller."""
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    uw.lander.linearVelocity = (0.0, 0.0)
    uw.lander.angularVelocity = 0.0
    ctrl = KTOController(env, time_budget=5.0)
    total_reward, done = 0.0, False
    while not done:
        action = ctrl.step(env)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        done = term or trunc
    landed = not uw.game_over and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
    return total_reward, landed


def rollout_pipeline(env, seed):
    """Run one episode through GuidanceController → DiffusionController."""
    obs, _ = env.reset(seed=seed)
    gc = GuidanceController(env, time_budget=5.0)
    total_reward, done = 0.0, False
    while not done:
        at = gc.step(env, obs)
        state = obs_to_lander_state(obs)
        spline = DiffusionController(
            timeout=TIMEOUT,
            t_obs_cmd_latency=DT,
            obstacles=[],
            lander_state=state,
            waypoint_goals=[],
            guidance_actions=[at],
            classifier_free_guidance=[],
            action_horizon=0.1,
            target_frequency=50.0,
        )
        tv, th = spline(DT)
        action = np.array([tv, th], dtype=np.float32)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        done = term or trunc
    uw = env.unwrapped
    landed = not uw.game_over and (uw.legs[0].ground_contact or uw.legs[1].ground_contact)
    return total_reward, landed


def test_guidance_clamping():
    """Guidance with tight margin clamps model output to target values."""
    at = ActionTarget(
        thrust_v=0.5, thrust_v_margin=0.001,
        thrust_h=-0.3, thrust_h_margin=0.001,
        thrust_t=DT, thrust_t_margin=0.001,
    )
    state = LanderState(
        t_sim_lander=0.0,
        q=(0.0, 0.0, 0.0),
        q_prime=(0.0, 0.0, 0.0),
        thrust=(0.0, 0.0),
        contacts=(False, False, False),
    )

    for _ in range(10):
        spline = DiffusionController(
            timeout=10.0,
            t_obs_cmd_latency=DT,
            obstacles=[],
            lander_state=state,
            waypoint_goals=[],
            guidance_actions=[at],
            classifier_free_guidance=[],
            action_horizon=0.1,
            target_frequency=50.0,
        )
        tv, th = spline(DT)
        assert abs(tv - 0.5) <= 0.001 + 1e-9, f"thrust_v={tv}, expected ~0.5"
        assert abs(th - (-0.3)) <= 0.001 + 1e-9, f"thrust_h={th}, expected ~-0.3"


def test_pipeline_matches_kto():
    """Diffusion pipeline with tight guidance should match standalone KTO."""
    n = 20

    gym.register(id="LL-kto-test", entry_point="lunar_lander:LunarLander",
                 max_episode_steps=1000)
    gym.register(id="LL-pipe-test", entry_point="lunar_lander:LunarLander",
                 max_episode_steps=1000)

    env_k = gym.make("LL-kto-test", render_mode=None, continuous=True)
    env_p = gym.make("LL-pipe-test", render_mode=None, continuous=True)

    kto_rewards, kto_lands = [], []
    pipe_rewards, pipe_lands = [], []

    for i in range(n):
        r, landed = rollout_kto(env_k, i)
        kto_rewards.append(r)
        kto_lands.append(landed)

    for i in range(n):
        r, landed = rollout_pipeline(env_p, i)
        pipe_rewards.append(r)
        pipe_lands.append(landed)

    env_k.close()
    env_p.close()

    kto_rate = np.mean(kto_lands)
    pipe_rate = np.mean(pipe_lands)
    kto_mean = np.mean(kto_rewards)
    pipe_mean = np.mean(pipe_rewards)

    print(f"\n{'seed':>4}  {'KTO':>10} {'Pipeline':>10} {'delta':>8}")
    print("-" * 38)
    for i in range(n):
        d = abs(kto_rewards[i] - pipe_rewards[i])
        print(f"{i:4d}  {kto_rewards[i]:10.3f} {pipe_rewards[i]:10.3f} {d:8.3f}")

    print(f"\nKTO:      land_rate={kto_rate:.0%}  reward={kto_mean:.2f}")
    print(f"Pipeline: land_rate={pipe_rate:.0%}  reward={pipe_mean:.2f}")
    print(f"Gaps:     land_rate={abs(kto_rate - pipe_rate):.0%}  reward={abs(kto_mean - pipe_mean):.2f}")

    assert abs(kto_rate - pipe_rate) <= 0.20, (
        f"Landing rate gap too large: KTO={kto_rate:.0%} vs pipeline={pipe_rate:.0%}"
    )
    assert abs(kto_mean - pipe_mean) < 2.0, (
        f"Reward gap too large: KTO={kto_mean:.2f} vs pipeline={pipe_mean:.2f}"
    )


def test_keyboard_override():
    """Keyboard action should override KTO in the guidance controller."""
    gym.register(id="LL-kb-test", entry_point="lunar_lander:LunarLander",
                 max_episode_steps=1000)
    env = gym.make("LL-kb-test", render_mode=None, continuous=True)
    obs, _ = env.reset(seed=0)

    gc = GuidanceController(env, time_budget=5.0)

    # Active keyboard action (not idle [0, 0])
    kb = np.array([0.8, -0.6], dtype=np.float32)
    at = gc.step(env, obs, keyboard_action=kb)

    assert abs(at.thrust_v - 0.8) < 1e-6, f"Expected thrust_v=0.8, got {at.thrust_v}"
    assert abs(at.thrust_h - (-0.6)) < 1e-6, f"Expected thrust_h=-0.6, got {at.thrust_h}"

    # Idle keyboard action should fall through to KTO
    kb_idle = np.array([0.0, 0.0], dtype=np.float32)
    at2 = gc.step(env, obs, keyboard_action=kb_idle)
    # Should NOT be [0, 0] — KTO should provide a real action
    assert not (at2.thrust_v == 0.0 and at2.thrust_h == 0.0), (
        "Idle keyboard should not override KTO"
    )

    env.close()


if __name__ == "__main__":
    test_guidance_clamping()
    print("test_guidance_clamping PASSED")

    test_keyboard_override()
    print("test_keyboard_override PASSED")

    test_pipeline_matches_kto()
    print("test_pipeline_matches_kto PASSED")

    print("\nALL PASSED")
