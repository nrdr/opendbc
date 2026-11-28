"""Pure tests for the Civic Experimental PCM pedal-like law (no car interface needed)."""
import math

from opendbc.car.honda import civic_pcm
from opendbc.car.honda.civic_pcm import PedalLaw, accel_ceiling, road_load_fraction

SPEED = 25.0  # m/s, inside the full-weight range


def fresh(accel):
  """A law whose acceleration filter starts at the measured value, as it does after the controller's inactive call."""
  law = PedalLaw()
  law.reset(accel)
  return law


def settle(law, request, accel, speed=SPEED, ticks=300, **kwargs):
  out = None
  for _ in range(ticks):
    out = law.update(request, accel, speed, True, **kwargs)
  return out


def test_road_load_is_the_interceptor_wind_brake_curve_at_factor_one():
  assert road_load_fraction(0.0) == 0.001
  assert road_load_fraction(2.3) == 0.002
  assert road_load_fraction(35.0) == 0.15
  assert road_load_fraction(60.0) == 0.15
  assert math.isclose(road_load_fraction(28.0) * 4.8, 0.568, abs_tol=0.005)
  assert all(road_load_fraction(v + 1.0) >= road_load_fraction(v) for v in range(60))
  assert road_load_fraction(float("nan")) == 0.0


def test_inactive_returns_no_authority_and_clears_state():
  law = PedalLaw()
  settle(law, 0.3, 0.1)
  assert law.integral > 0.0
  assert law.update(0.3, 0.1, SPEED, False) == (0.0, 0.0, 0.0, False)
  assert law.integral == 0.0 and not law.shedding and law.accel == 0.1


def test_nonfinite_input_fails_closed():
  law = PedalLaw()
  for bad in (float("nan"), float("inf")):
    assert law.update(bad, 0.0, SPEED, True) == (0.0, 0.0, 0.0, False)
    assert law.update(0.2, bad, SPEED, True) == (0.0, 0.0, 0.0, False)
    assert law.update(0.2, 0.0, bad, True) == (0.0, 0.0, 0.0, False)


def test_driving_holds_the_acceleration_mode_lead_and_scales_gas_with_the_request():
  law = fresh(0.3)
  lead, gas, weight, release = settle(law, 0.3, 0.3)
  assert lead == civic_pcm.ACCEL_MODE_LEAD and weight == 1.0 and not release
  assert math.isclose(gas, 198 * 0.3 / accel_ceiling(SPEED), rel_tol=0.02)
  low = settle(fresh(0.1), 0.1, 0.1)[1]
  assert low < gas / 2.5 and low < 0.25 * 198


def test_request_beyond_the_ceiling_saturates_without_winding_up():
  law = PedalLaw()
  lead, gas, _, _ = settle(law, 2.0, 0.2, ticks=3000)
  assert gas == 198 and lead == civic_pcm.ACCEL_MODE_LEAD
  assert law.integral == 0.0


def test_overshoot_sheds_inside_the_hold_band():
  law = PedalLaw()
  lead, gas, _, release = settle(law, 0.1, 0.4)
  assert law.shedding and not release
  assert math.isclose(lead, civic_pcm.SHED_GAIN * (0.1 - 0.4), rel_tol=0.02)
  assert gas > 0.0
  lead = settle(law, 0.0, 3.0)[0]
  assert lead == civic_pcm.SHED_LEAD[0] and lead > -9.0 / 3.6


def test_negative_request_closes_the_gas_and_only_a_material_slowdown_releases():
  law = PedalLaw()
  lead, gas, _, release = settle(law, -0.2, 0.0)
  assert gas == 0.0 and lead < 0.0 and not release and law.shedding
  assert not law.update(-0.49, 0.0, SPEED, True)[3]
  assert law.update(-0.5, 0.0, SPEED, True)[3]


def test_shedding_ends_only_when_acceleration_is_back_under_the_request():
  law = PedalLaw()
  settle(law, 0.2, 0.3)
  assert law.shedding
  settle(law, 0.2, 0.22)
  assert law.shedding
  lead = settle(law, 0.2, 0.19)[0]
  assert not law.shedding and lead == civic_pcm.ACCEL_MODE_LEAD
  settle(law, 0.2, 0.24)
  assert not law.shedding


def test_integral_is_clamped_and_frozen_while_braking_or_shedding():
  law = PedalLaw()
  settle(law, 0.3, 0.2, ticks=20000)
  assert law.integral == civic_pcm.INTEGRAL_LIMIT
  settle(law, 0.3, 0.34, ticks=20000)
  assert law.integral == -civic_pcm.INTEGRAL_LIMIT
  law = PedalLaw()
  settle(law, 0.3, 0.2, ticks=400)
  held = law.integral
  settle(law, 0.3, 0.2, ticks=400, braking=True)
  assert law.integral == held
  settle(law, 0.3, 0.6, ticks=100)
  assert law.shedding
  held = law.integral
  settle(law, 0.3, 0.6, ticks=400)
  assert law.integral == held


def test_previous_law_keeps_low_speed():
  law = PedalLaw()
  assert law.update(0.2, 0.0, 6.0, True)[2] == 0.0
  assert law.update(0.2, 0.0, 8.0, True)[2] == 0.5
  assert law.update(0.2, 0.0, 10.0, True)[2] == 1.0
  slow = PedalLaw()
  settle(slow, 0.3, 0.1, speed=5.0)
  assert slow.integral == 0.0


def test_a_gap_in_the_callers_frame_counter_restarts_the_state():
  law = PedalLaw()
  for tick in range(400):
    law.update(0.3, 0.2, SPEED, True, tick=tick)
  assert law.integral > 0.0
  law.update(0.3, 0.2, SPEED, True, tick=400)
  assert law.integral > 0.0
  law.update(0.3, 0.5, SPEED, True, tick=900)
  assert law.accel > 0.45 and law.integral <= 0.0


def test_speed_error_term_is_proportional_bounded_and_shapes_propulsion_only():
  ceiling = accel_ceiling(SPEED)
  # 0.4 m/s under the planner's speed adds 0.1 m/s2 to the command
  gas = fresh(0.1).update(0.1, 0.1, SPEED, True, reference=SPEED + 0.4)[1]
  assert math.isclose(gas, 198 * 0.2 / ceiling, rel_tol=0.02)
  assert gas > fresh(0.1).update(0.1, 0.1, SPEED, True, reference=SPEED)[1]
  # bounded
  gas = fresh(0.0).update(0.0, 0.0, SPEED, True, reference=SPEED + 10.0)[1]
  assert math.isclose(gas, 198 * civic_pcm.SPEED_ERROR_LIMIT / ceiling, rel_tol=0.02)
  # over the planner's speed the throttle is shed, but nothing is released or braked by this term
  law = fresh(0.0)
  lead, gas, _, release = law.update(0.05, 0.0, SPEED, True, reference=SPEED - 1.0)
  assert law.shedding and gas == 0.0 and lead < 0.0 and not release
  # the material-slowdown release looks at the raw request only
  assert fresh(0.0).update(-0.5, 0.0, SPEED, True, reference=SPEED + 10.0)[3]
  assert not fresh(0.0).update(-0.3, 0.0, SPEED, True, reference=SPEED - 10.0)[3]
  # a missing or non-finite reference contributes nothing
  base = fresh(0.1).update(0.1, 0.1, SPEED, True)[1]
  assert fresh(0.1).update(0.1, 0.1, SPEED, True, reference=None)[1] == base
  assert fresh(0.1).update(0.1, 0.1, SPEED, True, reference=float("nan"))[1] == base
