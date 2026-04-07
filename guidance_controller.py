"""GuidanceController — wraps KTOController + keyboard into ActionTargets."""

import numpy as np

from lunar_lander import KTOController, DT
from diffusion_controller import ActionTarget


GUIDANCE_MARGIN = 0.1


class GuidanceController:
    """Thin wrapper: KTO tracking controller with keyboard override.

    Produces ActionTarget objects suitable for DiffusionController's
    guidance_actions input.
    """

    def __init__(self, env, **kto_kwargs):
        self.kto = KTOController(env, **kto_kwargs)

    def step(self, env, obs, keyboard_action=None):
        """Return an ActionTarget for this step.

        If keyboard_action is actively pressed (not idle [-1, 0]),
        use it instead of the KTO controller.
        """
        if keyboard_action is not None and _has_active_keys(keyboard_action):
            thrust_v = float(keyboard_action[0])
            thrust_h = float(keyboard_action[1])
        else:
            action = self.kto.step(env)
            thrust_v = float(action[0])
            thrust_h = float(action[1])

        return ActionTarget(
            thrust_v=thrust_v,
            thrust_v_margin=GUIDANCE_MARGIN,
            thrust_h=thrust_h,
            thrust_h_margin=GUIDANCE_MARGIN,
            thrust_t=DT,
            thrust_t_margin=GUIDANCE_MARGIN,
        )


def _has_active_keys(keyboard_action):
    """Check if the keyboard action differs from idle state [-1, 0]."""
    return not (keyboard_action[0] == 0.0 and keyboard_action[1] == 0.0)
