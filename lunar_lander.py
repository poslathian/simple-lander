"""Keyboard-controlled LunarLander-v3 (continuous action space).

Controls:
    Up / W      — main engine
    Left / A    — left orientation engine
    Right / D   — right orientation engine
    Q / Esc     — quit

Hold multiple keys together (e.g. Up + Left) to fire main + side engines simultaneously.
"""

import gymnasium as gym
import numpy as np
import pygame


def main():
    env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    env.reset()

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

        keys = pygame.key.get_pressed()
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

        main_throttle = 1.0 if up else -1.0
        side = (-1.0 if left else 0.0) + (1.0 if right else 0.0)
        action = np.array([main_throttle, side], dtype=np.float32)

        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()

        clock.tick(50)

    env.close()


if __name__ == "__main__":
    main()
