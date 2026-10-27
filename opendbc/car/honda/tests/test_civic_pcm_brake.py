"""Civic brake dead-zone recovery, isolation, and deliberately bounded toy plants.

These plants check feedback logic, not real vehicle dynamics or road safety.
"""
from collections import deque
import math

import numpy as np
import pytest

from opendbc.car.honda.civic_pcm_brake import CivicPCMBrakeFeedback
from opendbc.car.honda.tests.test_civic_pcm import interface, control
from opendbc.car.honda.tests.test_civic_pcm_speed_control import START, reference
from opendbc.car.honda.values import CAR


def setup(*, wind=2.9863840730443507, interceptor=False, car=CAR.HONDA_CIVIC):
  ci, config = interface(car, interceptor=interceptor)
  ci.CS.out.canValid = True
  ci.CS.car_gas_available = True
  ci.CS.car_gas = 0.
  ci.CC.windfactor = wind
  ci.CC.nidec_brake_pid.i = 2.179123328066013
  return ci, config


def step(ci, requested, tick, *, speed=23., actual=.15, pitch=-.05, ref=22.):
  now = START + tick * 10_000_000
  ci.CS.out.vEgo = ci.CS.out.vEgoRaw = float(speed)
  ci.CS.out.aEgo = float(actual)
  cc = control(requested)
  cc.orientationNED = [0., float(pitch), 0.]
  return ci.apply(cc.as_reader(), reference(ref, now), now)[1]


def test_feedback_can_start_when_base_command_is_zero():
  pid = CivicPCMBrakeFeedback()
  values = [pid.update(-.7, .15, 0., allowed=True, saturated=False, maximum=255/256, dt=.02) for _ in range(100)]
  assert values[0] > 0
  assert max(np.diff([0., *values])) <= pid.RISE_RATE * .02 + 1e-12
  assert values[-1] > .2


def test_base_response_is_not_slew_limited_when_more_deceleration_is_needed():
  pid = CivicPCMBrakeFeedback()
  assert pid.update(-2., 0., .8, allowed=True, saturated=False, maximum=255/256, dt=.02) >= .8


def test_feedback_can_reduce_braking_that_exceeds_the_requested_deceleration():
  pid = CivicPCMBrakeFeedback()
  assert pid.update(-.7, -2., .8, allowed=True, saturated=False, maximum=255/256, dt=.02) < .8


def test_negative_trim_cannot_block_braking_after_feedforward_falls_to_zero():
  pid = CivicPCMBrakeFeedback()
  for _ in range(100):
    pid.update(-.7, -2., .8, allowed=True, saturated=False, maximum=255/256, dt=.02)
  assert pid.integral < 0
  assert pid.update(-.7, .2, 0., allowed=True, saturated=False, maximum=255/256, dt=.02) > 0


def test_feedback_unwinds_at_saturation_and_resets_without_authority():
  pid = CivicPCMBrakeFeedback()
  for _ in range(500):
    pid.update(-.7, .2, 1., allowed=True, saturated=True, maximum=255/256, dt=.02)
  assert pid.integral == 0
  for _ in range(100):
    pid.update(-.7, .2, 0., allowed=True, saturated=False, maximum=255/256, dt=.02)
  before = pid.integral
  pid.update(-.7, -2., 1., allowed=True, saturated=True, maximum=255/256, dt=.02)
  assert pid.integral < before
  assert pid.update(0., 1., 0., allowed=False, saturated=False, maximum=255/256, dt=.02) == 0
  assert pid.integral == pid.feedback == 0


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_invalid_acceleration_cannot_retain_feedback(bad):
  pid = CivicPCMBrakeFeedback()
  pid.integral = pid.feedback = .5
  assert pid.update(0., bad, 0., allowed=True, saturated=False, maximum=255/256, dt=.02) == 0
  assert pid.integral == pid.feedback == 0


@pytest.mark.parametrize('shaping', [False, True])
def test_recorded_downhill_calibration_no_longer_suppresses_braking(shaping):
  from dataclasses import replace
  ci, config = setup()
  config.honda.provider.live_tuning = replace(config.honda.provider.live_tuning, ecu_matched_long=shaping)
  first = None
  for tick in range(400):
    messages = step(ci, -.7, tick)
    if ci.CC.apply_brake_last and first is None:
      first = tick / 100
    for address, payload, _ in messages:
      if address == 0x30C and ci.CC.apply_brake_last:
        assert payload[2] == 0 and payload[:2] == b'\x00\x00'
    assert ci.CC.accel == pytest.approx(-.7)
    assert ci.CC.nidec_pid_factor == 0.
  assert first is not None and first < .2


@pytest.mark.parametrize('requested', [0., .1])
def test_downhill_speed_hold_can_require_braking_even_with_nonnegative_target_acceleration(requested):
  ci, _ = setup()
  for tick in range(200):
    step(ci, requested, tick, actual=.7, pitch=-.08)
  assert ci.CC.apply_brake_last > 0
  assert ci.CC.accel == pytest.approx(requested)


def test_flat_positive_propulsion_cannot_inherit_brake_feedback():
  ci, _ = setup()
  for tick in range(100):
    step(ci, -.7, tick)
  assert ci.CC.civic_brake_feedback.integral > 0
  for tick in range(100, 200):
    step(ci, .5, tick, actual=1., pitch=0., ref=23.)
  assert ci.CC.civic_brake_feedback.integral == ci.CC.civic_brake_feedback.feedback == 0
  assert ci.CC.apply_brake_last == 0
  assert ci.CC.pcm_gas_sent == 153  # unchanged acceleration-overshoot allowance
  assert ci.CC.pcm_speed_sent > ci.CS.out.vEgo


@pytest.mark.parametrize('reason', ['driver_gas', 'driver_brake', 'aeb', 'invalid_can'])
def test_driver_override_and_invalid_data_clear_new_feedback(reason):
  ci, _ = setup()
  for tick in range(100):
    step(ci, -.7, tick)
  if reason == 'driver_gas':
    ci.CS.out.gasPressed = True
  elif reason == 'driver_brake':
    ci.CS.out.brakePressed = True
  elif reason == 'aeb':
    ci.CS.out.stockAeb = True
  elif reason == 'invalid_can':
    ci.CS.out.canValid = False
  for tick in range(100, 104):
    step(ci, -.7, tick)
  assert ci.CC.civic_brake_feedback.integral == ci.CC.civic_brake_feedback.feedback == 0


@pytest.mark.parametrize('car,interceptor', [(CAR.HONDA_CLARITY, False), (CAR.HONDA_CLARITY, True),
                                           (CAR.HONDA_CIVIC, True), (CAR.HONDA_CIVIC_BOSCH, False)])
def test_other_platforms_never_call_civic_brake_feedback(car, interceptor):
  ci, _ = setup(car=car, interceptor=interceptor)

  def unexpected(*args, **kwargs):
    pytest.fail('Out-of-scope brake path entered Civic feedback')
  ci.CC.civic_brake_feedback.update = unexpected
  for tick in range(100):
    step(ci, -.7, tick)


def simulate(gain, delay, tau, wind, requested, *, seconds=25., grade_change=False):
  ci, _ = setup(wind=wind)
  ticks = round(seconds * 100)
  commands = deque([0.] * (round(delay * 100) + 1))
  applied = 0.
  speed = ref = 27.
  measured = actual = .35
  errors, efforts, accelerations = [], [], []
  for tick in range(ticks):
    coast = .35 if not grade_change or tick < ticks//2 else .05
    pitch = -math.asin((coast + .25) / 9.81)
    # This isolates coasting/braking; measured ECU propulsion is zero as in the
    # confirmed events. It is not a model of Honda's proprietary throttle servo.
    step(ci, requested, tick, speed=speed, actual=measured, pitch=pitch, ref=ref)
    command = ci.CC.apply_brake_last / ci.CC.params.NIDEC_BRAKE_MAX
    commands.append(command)
    delayed = commands.popleft()
    applied += (delayed - applied) * min(.01 / tau, 1.)
    actual = coast - gain * applied
    measured += (actual - measured) * .01 / .10
    speed = max(speed + actual * .01, 0.)
    ref = max(ref + requested * .01, 0.)
    errors.append(actual - requested)
    efforts.append(command)
    accelerations.append(actual)
  return np.array(errors), np.array(efforts), np.array(accelerations)


@pytest.mark.parametrize('gain', [2., 3.7, 6.])
@pytest.mark.parametrize('delay,tau', [(0., .1), (.2, .3), (.5, .6)])
@pytest.mark.parametrize('wind', [.1, 1., 2.9863840730443507])
@pytest.mark.parametrize('requested', [-.7, 0.])
def test_coast_only_closed_loop_recovers_requested_acceleration(gain, delay, tau, wind, requested):
  errors, efforts, actual = simulate(gain, delay, tau, wind, requested)
  assert abs(errors[-500:].sum() / 500) < .1
  assert np.std(errors[-500:]) < .12
  assert np.isfinite(actual).all()
  assert np.min(efforts) >= 0 and np.max(efforts) <= 255/256


def test_grade_reduction_does_not_retain_old_braking_indefinitely():
  errors, efforts, _ = simulate(3.7, .2, .3, 2.9863840730443507, -.7, seconds=30., grade_change=True)
  assert abs(errors[-500:].sum() / 500) < .1
  assert efforts[-500:].sum() / 500 < efforts[1000:1400].sum() / 400
