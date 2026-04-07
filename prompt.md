# Position Diffusion Controller

## Goal

Create a much simpler version of diffusion_controller.pyi and its implementation
that achieves faster training and better control stability.

## Simplifications

1. Edit the DiffusionController.pyi so that we have just:
   - A single waypoint (not a list of 5)
   - A single guidance action (not a list of 5)
   - A single outcome: success(1), fail(-1), don't know(0)

## Context

The v0.1.0 codebase uses thrust B-splines (15 CP x 2) which fail closed-loop
despite near-perfect open-loop reconstruction. Prior experiments showed that
position B-splines (10 CP x 3: x, y, theta) with PD tracking work but need
a bigger model (hidden=512+).

This branch ("position") simplifies the interface to reduce conditioning
dimensionality, making the model easier to train and the whole pipeline
easier to reason about.
