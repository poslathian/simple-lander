"""Box2D-matched physics constants for the simplified lunar lander."""

SCALE = 30.0
FPS = 50
DT = 1.0 / FPS
GRAVITY = -10.0

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 1.5

MAIN_ENGINE_Y_LOCATION = 4
SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12

# Computed from Box2D polygon (density=5.0) + two legs (density=1.0)
SYSTEM_MASS = 4.958889
LANDER_INERTIA_CM = 0.783881
LANDER_CM_LOCAL = (0.0, 0.101307)
