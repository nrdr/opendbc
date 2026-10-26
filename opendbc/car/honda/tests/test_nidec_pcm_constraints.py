"""PCM-only comfort shaping, real CAN packing, and braking isolation."""
from dataclasses import replace

import numpy as np
import pytest

from opendbc.car import DT_CTRL
from opendbc.car.honda.nidec_pcm_constraints import (
  ACCEL_RAMP_DOWN, ACCEL_RAMP_UP, NidecPCMConstraints,
  neutral_accel_deadband, taper_positive_accel,
)
from opendbc.car.honda.tests.test_civic_pcm import control, interface
from opendbc.car.honda.tests.test_civic_pcm_speed_control import START, reference
from opendbc.car.honda.tests.test_mvl_nrdr_integration import learned_state
from opendbc.car.honda.values import CAR, DBC
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP


def setup(car=CAR.HONDA_CIVIC, *, enabled=True, interceptor=False, stock=False, learning=False):
  ci, config = interface(car, interceptor=interceptor, learning=learning)
  if stock:
    # Rebuild after selecting the mode; platform scope is fixed at construction.
    ci.CP_SP.flags |= HondaFlagsSP.STOCK_LONGITUDINAL
    from opendbc.car.honda.carcontroller import CarController
    ci.CC = CarController(DBC[car], ci.CP, ci.CP_SP, config.honda)
  config.honda.provider.live_tuning = replace(config.honda.provider.live_tuning, ecu_matched_long=enabled)
  return ci, config


def step(ci, accel, tick, *, active=True, speed=20.0, fresh=True):
  now = START + tick * 10_000_000
  cc = control(accel)
  cc.enabled = cc.longActive = active
  sp = reference(speed, now if fresh else now - 300_000_000)
  return ci.apply(cc.as_reader(), sp, now)[1]


@pytest.mark.parametrize('speed,expected', [(0., .384), (2.5, .384), (10., .288), (20., .144), (30., .024), (40., .024)])
def test_pilot_deadband_keeps_original_normalized_command_units(speed, expected):
  assert neutral_accel_deadband(speed) == pytest.approx(expected)


@pytest.mark.parametrize('speed', [0., 2.5, 10., 20., 30., 40.])
def test_deadband_has_no_hard_step_and_preserves_full_authority(speed):
  knee = neutral_accel_deadband(speed)
  assert taper_positive_accel(knee, speed) == 0.
  assert taper_positive_accel(2., speed) == 2.
  values = [taper_positive_accel(float(a), speed) for a in np.linspace(0., 3., 6001)]
  assert np.diff(values).min() >= -1e-12
  assert np.diff(values).max() < .00062
  assert taper_positive_accel(2., speed) == 2.


def test_ramp_limits_do_not_reduce_final_acceleration():
  shape = NidecPCMConstraints()
  values = [shape.update(2., 20., enabled=True, active=True) for _ in range(60)]
  assert values[0] == pytest.approx(ACCEL_RAMP_UP * DT_CTRL)
  assert np.diff(values).max() <= ACCEL_RAMP_UP * DT_CTRL + 1e-12
  assert values[-1] == 2.
  falling = [shape.update(.1, 20., enabled=True, active=True) for _ in range(60)]
  assert np.diff([2., *falling]).min() >= -ACCEL_RAMP_DOWN * DT_CTRL - 1e-12
  assert falling[-1] == 0.


@pytest.mark.parametrize('accel', [0., -.01, -.1, -.5, -1., -4.])
def test_negative_and_zero_requests_are_never_delayed(accel):
  shape = NidecPCMConstraints()
  for _ in range(100):
    shape.update(2., 20., enabled=True, active=True)
  assert shape.update(accel, 20., enabled=True, active=True) == accel


def test_neutral_zone_replaces_the_old_timed_gas_brake_blackout():
  shape = NidecPCMConstraints()
  assert shape.update(-.1, 20., enabled=True, active=True) == -.1
  assert shape.update(.8, 20., enabled=True, active=True) == pytest.approx(ACCEL_RAMP_UP * DT_CTRL)
  assert shape.update(-3., 20., enabled=True, active=True) == -3.


@pytest.mark.parametrize('reason', ['disabled', 'inactive', 'launch'])
def test_bypass_is_exact_and_does_not_retain_old_filter_state(reason):
  shape = NidecPCMConstraints()
  args = dict(enabled=True, active=True, launch=False)
  for _ in range(50):
    shape.update(.8, 20., **args)
  args[{'disabled': 'enabled', 'inactive': 'active'}.get(reason, reason)] = reason not in ('disabled', 'inactive')
  assert shape.update(.1, 20., **args) == .1


@pytest.mark.parametrize('car,interceptor,stock,expected', [
  (CAR.HONDA_CIVIC, False, False, True),
  (CAR.HONDA_PILOT, False, False, True),
  (CAR.HONDA_CLARITY, False, False, True),
  (CAR.HONDA_CIVIC, True, False, False),
  (CAR.HONDA_CLARITY, True, False, False),
  (CAR.HONDA_CIVIC_BOSCH, False, False, False),
  (CAR.HONDA_CIVIC, False, True, False),
])
def test_exact_pcm_scope(car, interceptor, stock, expected):
  ci, _ = setup(car, interceptor=interceptor, stock=stock)
  assert ci.CC.nidec_pcm is expected


def test_toggle_reaches_actual_civic_pcm_packets_and_does_not_zero_allowance():
  off, _ = setup(enabled=False)
  on, _ = setup(enabled=True)
  for tick in range(201):
    a, b = step(off, .2, tick), step(on, .2, tick)
  assert on.CC.pcm_speed_sent < off.CC.pcm_speed_sent
  assert on.CC.pcm_speed_sent == pytest.approx(20.0 + ((.2 - .144) * 2. / (2. - .144)) * 4.)
  assert on.CC.pcm_gas_sent == off.CC.pcm_gas_sent == 198
  assert all(dat[2] > 0 for addr, dat, _ in b if addr == 0x30C)


@pytest.mark.parametrize('car,interceptor,stock', [
  (CAR.HONDA_CIVIC, True, False), (CAR.HONDA_CLARITY, True, False),
  (CAR.HONDA_CIVIC_BOSCH, False, False), (CAR.HONDA_CIVIC, False, True),
])
def test_toggle_is_packet_identical_outside_pcm(car, interceptor, stock):
  off, _ = setup(car, enabled=False, interceptor=interceptor, stock=stock)
  on, _ = setup(car, enabled=True, interceptor=interceptor, stock=stock)
  for tick in range(500):
    accel = [1., .1, -.1, -2., .8][tick // 100]
    assert step(off, accel, tick) == step(on, accel, tick)


@pytest.mark.parametrize('car', [CAR.HONDA_CIVIC, CAR.HONDA_PILOT, CAR.HONDA_CLARITY])
def test_all_brake_packets_unchanged_with_pcm_constraints(car):
  off, _ = setup(car, enabled=False)
  on, _ = setup(car, enabled=True)
  for tick in range(600):
    accel = [1., .2, -.1, -3., .8, -4.][tick // 100]
    a, b = step(off, accel, tick), step(on, accel, tick)
    assert [(addr, dat, bus) for addr, dat, bus in a if addr != 0x30C] == [
      (addr, dat, bus) for addr, dat, bus in b if addr != 0x30C]
    assert on.CC.apply_brake_last == off.CC.apply_brake_last
    if on.CC.apply_brake_last:
      assert all(dat[2] == 0 for addr, dat, _ in b if addr == 0x30C)


@pytest.mark.parametrize('reason', ['disengage', 'driver_brake', 'aeb', 'stale'])
def test_civic_safety_release_still_wins_over_shaping(reason):
  ci, _ = setup()
  for tick in range(100):
    step(ci, 1., tick)
  if reason == 'driver_brake':
    ci.CS.out.brakePressed = True
  elif reason == 'aeb':
    ci.CS.out.stockAeb = True
  messages = step(ci, 1., 100, active=reason != 'disengage', fresh=reason != 'stale')
  assert [dat[2] for addr, dat, _ in messages if addr == 0x30C] == [0]


def test_pcm_learner_does_not_retrain_for_shaped_commands():
  ci, config = setup(CAR.HONDA_PILOT, learning=True)
  before = learned_state(ci.CC)
  for tick in range(6100):
    step(ci, .5, tick)
  assert learned_state(ci.CC) == before
  assert all(all(key.startswith('lat_accel_factor_') for key in item) for item in config.honda.provider.persisted)


@pytest.mark.parametrize('car', [CAR.HONDA_PILOT, CAR.HONDA_CLARITY])
def test_toggle_off_waits_for_shaped_samples_to_leave_learner_history(car):
  from opendbc.car.honda.carcontroller import NIDEC_SERVO_LAG_FRAMES
  ci, config = setup(car, learning=True)
  for tick in range(300):
    step(ci, .5, tick)
  assert not any(eligible for _, _, eligible in ci.CC.speed_cmd_hist)
  before = learned_state(ci.CC)
  config.honda.provider.live_tuning = replace(config.honda.provider.live_tuning, ecu_matched_long=False)
  for tick in range(300, 300 + NIDEC_SERVO_LAG_FRAMES):
    step(ci, .5, tick)
    assert learned_state(ci.CC) == before
  assert ci.CC.pcm_constraints_washout == 0
  assert all(eligible for _, _, eligible in ci.CC.speed_cmd_hist)


@pytest.mark.parametrize('car', [CAR.HONDA_PILOT, CAR.HONDA_CLARITY])
def test_generic_pcm_positive_commands_are_shaped_but_full_acceleration_returns(car):
  off, _ = setup(car, enabled=False)
  on, _ = setup(car, enabled=True)
  for tick in range(201):
    a, b = step(off, .2, tick), step(on, .2, tick)
  def speed(messages):
    return [int.from_bytes(dat[:2], 'big') for addr, dat, _ in messages if addr == 0x30C]
  assert speed(b)[0] < speed(a)[0]
  for tick in range(201, 401):
    a, b = step(off, 2., tick), step(on, 2., tick)
  assert a == b


def test_grade_brake_zero_crossing_cannot_bypass_or_reset_propulsion_filter():
  ci, _ = setup()
  speeds = []
  for tick in range(400):
    now = START + tick * 10_000_000
    cc = control(.7)
    # The downhill grade term alternates around the raw braking boundary.
    # The existing brake hysteresis keeps actual brake packets at zero.
    cc.orientationNED = [0., -.075 if tick % 20 < 10 else -.070, 0.]
    _, packets = ci.apply(cc.as_reader(), reference(20., now), now)
    for address, payload, _ in packets:
      if address == 0x30C and tick >= 100:
        speeds.append(int.from_bytes(payload[:2], 'big') * .01 / 3.6)
        assert ci.CC.apply_brake_last == 0
  assert max(speeds) - min(speeds) < .01


def test_shaped_mode_clears_pending_ceiling_evidence_before_learning_resumes():
  from opendbc.car.honda.carcontroller import NIDEC_SERVO_LAG_FRAMES
  ci, config = setup(CAR.HONDA_PILOT, learning=True)
  ci.CC.sat_excess_frames = 101
  ci.CC.sat_deficit_frames = 151
  for tick in range(300):
    step(ci, .5, tick)
    assert ci.CC.sat_excess_frames == ci.CC.sat_deficit_frames == 0
  config.honda.provider.live_tuning = replace(config.honda.provider.live_tuning, ecu_matched_long=False)
  for tick in range(300, 300 + NIDEC_SERVO_LAG_FRAMES):
    step(ci, .5, tick)
    assert ci.CC.sat_excess_frames == ci.CC.sat_deficit_frames == 0
  before = ci.CC.sat_accel
  ci.CS.out.aEgo = ci.CC.a_ego_filt = 3.
  step(ci, .5, 300 + NIDEC_SERVO_LAG_FRAMES)
  assert ci.CC.sat_excess_frames <= 1 and ci.CC.sat_deficit_frames <= 1
  assert ci.CC.sat_accel == before
