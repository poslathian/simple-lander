# Test Plan

Tests to catch all issues identified in issues.md, plus correctness tests for core components.

## Test 1: B-spline knot vector consistency

**Issue:** Knot vector mismatch between `_fit_kto_window()` (DEGREE+1 boundary repeats) and `_make_position_spline()` (DEGREE repeats).

**Test:** Create known CPs, build splines with both knot constructions, verify they produce different curves. Then verify that `_make_position_spline` with corrected knots matches `_fit_kto_window`'s knot structure.

**Implementation:** `test_knot_vector_consistency()` — construct CPs, build both splines, evaluate at several points, assert match after fix.

## Test 2: _make_position_spline clamping

**Issue:** Spline with only DEGREE boundary knots isn't properly clamped — first/last CPs don't pin endpoints.

**Test:** Build a position spline from known CPs. Evaluate at t=0 and t=duration. The values should equal the first and last CPs respectively (clamped property).

**Implementation:** `test_position_spline_clamped()` — create CPs with distinct first/last values, build spline, check endpoints.

## Test 3: Conditioning q_prev correctness in collect.py

**Issue:** q_prev always equals q_now in collect.py because it reads `_last_inference_q` after `inference()` updates it.

**Test:** Run a short collection episode, extract the conditioning vectors, verify that q_prev (indices 4:7) differs from q_now (indices 1:4) after the first inference call.

**Implementation:** `test_collect_qprev_differs_from_qnow()` — run 2+ inference calls, check conditioning.

## Test 4: Tracking controller inverse dynamics accuracy

**Issue:** `_tracking_step()` omits the `1/cos(2θ)` factor, causing thrust errors at large angles.

**Test:** At several θ values, compute the thrust from `_tracking_step()` and from the full Cramer's rule formula. Compare the error.

**Implementation:** `test_tracking_inverse_dynamics_accuracy()` — parametrize over θ values, verify the tracking formula matches full inverse dynamics within tolerance.

## Test 5: guidance_controller.py import check

**Issue:** `guidance_controller.py` imports `ActionTarget` from `diffusion_controller`, which doesn't exist.

**Test:** Attempt to import `guidance_controller` and catch the ImportError.

**Implementation:** `test_guidance_controller_imports()` — try import, expect failure (or success after fix).

## Test 6: test_diffusion_pipeline.py import check

**Issue:** `test_diffusion_pipeline.py` imports non-existent `DiffusionController` and `LanderState`.

**Test:** Attempt to import `test_diffusion_pipeline` and verify it either works or documents the failure.

**Implementation:** `test_diffusion_pipeline_imports()` — try import, expect failure (or success after fix).

## Test 7: NoiseModel output shape

**Test:** Verify `NoiseModel.predict()` returns (N_CPS, N_CHANNELS) shaped array.

**Implementation:** `test_noise_model_output_shape()` — simple shape check.

## Test 8: DDIMSampler output shape

**Test:** Verify DDIM sampling produces correct output dimensions.

**Implementation:** `test_ddim_output_shape()` — build model, sample, check shape.

## Test 9: _build_cond output

**Test:** Verify conditioning vector has correct dimension and structure.

**Implementation:** `test_build_cond_dimensions()` — call with known inputs, verify shape and values.

## Test 10: KTODiffusionController get_action returns zero after KTO exhausted

**Test:** After KTO plan is exhausted, `get_action()` should return (0, 0).

**Implementation:** `test_get_action_after_kto_exhausted()` — mock a controller past its KTO duration, verify zero output.

## Test 11: Margin blending in get_action

**Test:** At margin=0, get_action should track pure KTO. At margin=1, it should track pure diffusion.

**Implementation:** `test_margin_blending()` — verify the linear blend formula.

## Test 12: Per-step margin jitter

**Issue:** `dagger_loop.py` samples a different margin every sim step within an episode.

**Test:** Verify that rapid margin changes cause reference jitter. Measure the position reference variance over a short window.

**Implementation:** `test_margin_jitter_effect()` — run short episode, measure reference stability.

## Implementation Plan

1. Create `test_issues.py` with all tests above
2. Tests that verify current bugs should be written as `xfail` or as assertions that catch the bug (failing when bug exists, passing when fixed)
3. Use pytest fixtures for env creation
4. Tests should be fast (no KTO solves where possible, mock where appropriate)
5. Tests that require KTO solve should be marked `slow`
