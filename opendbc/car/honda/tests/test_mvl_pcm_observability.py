"""Exercise MVL's learner gates through nrdr's real Honda controller."""
from collections import deque
from dataclasses import replace

import pytest

from opendbc.car import structs
from opendbc.car.honda.carcontroller import NIDEC_SERVO_LAG_FRAMES
from opendbc.car.honda.tests.test_civic_pcm import control, interface
from opendbc.car.honda.tests.test_mvl_nrdr_integration import learned_state
from opendbc.car.honda.values import CAR
from opendbc.sunnypilot.car.tests.runtime_config import StaticHondaConfigProvider


def settled(car=CAR.HONDA_CRV, *, speed=20., factor=4., lagged=.7, learning=True, interceptor=False):
  ci, config = interface(car, learning=learning, interceptor=interceptor)
  ci.CS.out.vEgo = speed
  ci.CS.out.cruiseState.speed = 33.
  ci.CC.long_active_prev = True
  ci.CC.speed_cmd_hist = deque([(lagged, True, True)] * NIDEC_SERVO_LAG_FRAMES, maxlen=NIDEC_SERVO_LAG_FRAMES)
  ci.CC.speed_factors = dict.fromkeys(ci.CC.speed_factors, factor)
  return ci, config.honda.provider


def step(ci, accel=.7, tick=0, sp=None):
  return ci.apply(control(accel).as_reader(), sp if sp is not None else structs.CarControlSP(), tick * 10_000_000)


@pytest.mark.parametrize("speed,governed", [(30.99, False), (31., True), (32.4, True), (34., True)])
def test_set_speed_boundary_and_deficit_evidence(speed, governed):
  ci, _ = settled(speed=speed, factor=16.)
  ci.CC.sat_deficit_frames = 150  # one more eligible, maxed sample would lower the ceiling
  before = learned_state(ci.CC)
  step(ci)
  after = learned_state(ci.CC)
  assert ci.CC.sat_deficit_frames == (0 if governed else 151)
  assert (after["sat_accel"] == before["sat_accel"]) == governed
  assert (after["speed_factors"] == before["speed_factors"]) == governed
  if governed:
    assert after["speed_alphas"] == before["speed_alphas"]


@pytest.mark.parametrize("speed,governed", [(30.99, False), (31., True), (33., True)])
def test_governor_blocks_wind_and_speed_updates_even_with_unsaturated_lead(speed, governed):
  ci, _ = settled(speed=speed, lagged=.2)
  before = learned_state(ci.CC)
  step(ci)
  after = learned_state(ci.CC)
  for field in ("windfactor", "speed_alphas", "speed_factors"):
    assert (before[field] == after[field]) == governed


def test_leaving_governor_resumes_live_cap_without_reset_or_restart():
  ci, _ = settled(speed=32., factor=43.)
  step(ci)
  assert set(ci.CC.speed_factors.values()) == {43.}
  ci.CS.out.cruiseState.speed = 36.
  step(ci, tick=1)
  assert max(ci.CC.speed_factors.values()) == 16.


@pytest.mark.parametrize("factor,pinned", [(4., False), (5., False), (6., True)])
@pytest.mark.parametrize("measured", [-.4, -.6])
def test_lagged_coast_floor_allows_reductions_but_not_growth(factor, pinned, measured):
  # A -0.5 lagged command at factor 5 is exactly -2.5 m/s of lead, still observable.
  ci, _ = settled(factor=factor, lagged=-.5)
  ci.CS.out.aEgo = measured
  before = learned_state(ci.CC)
  step(ci, accel=-.5)
  for key, would_grow in (("speed_factors", measured > -.5), ("speed_alphas", measured < -.5)):
    deltas = [v - before[key][k] for k, v in getattr(ci.CC, key).items()]
    if would_grow and pinned:
      assert not any(deltas)
    elif would_grow:
      assert max(deltas) > 0
    else:
      assert min(deltas) < 0


@pytest.mark.parametrize("car,interceptor,learning,expect_cap", [
  (CAR.HONDA_CRV, False, True, True), (CAR.HONDA_CIVIC, False, True, True),
  (CAR.HONDA_CLARITY, False, True, False), (CAR.HONDA_CLARITY, True, True, False),
  (CAR.HONDA_CIVIC, True, True, False), (CAR.HONDA_CRV, False, False, False),
])
def test_saved_state_is_not_clipped_on_load_or_on_excluded_paths(monkeypatch, car, interceptor, learning, expect_cap):
  monkeypatch.setattr(StaticHondaConfigProvider, "load_mvl_param",
                      lambda self, key, default: 43. if key.startswith("speed_factor") else default)
  ci, _ = interface(car, interceptor=interceptor, learning=learning)
  assert set(ci.CC.speed_factors.values()) == {43.}
  # Rolling engagement has its own 200-frame recovery window, then a servo delay.
  for tick in range(200 + NIDEC_SERVO_LAG_FRAMES + 2):
    if expect_cap and tick < 200:
      assert set(ci.CC.speed_factors.values()) == {43.}
    step(ci, tick=tick)
  if expect_cap:
    assert max(ci.CC.speed_factors.values()) == 16.
  else:
    assert min(ci.CC.speed_factors.values()) > 16.


@pytest.mark.parametrize("factor,admitted", [(20., True), (99., True), (100., False)])
def test_civic_speed_reference_keeps_its_own_admission_bound(factor, admitted):
  ci, _ = settled(CAR.HONDA_CIVIC, factor=factor)
  before = learned_state(ci.CC)
  for tick in range(1, 101):
    sp = structs.CarControlSP(civicPcmEnabled=True, civicPcmReferenceSpeed=20., civicPcmReferenceMonoTime=tick * 10_000_000)
    step(ci, tick=tick, sp=sp)
  assert (ci.CC.pcm_speed_sent > ci.CS.out.vEgo) == admitted
  assert learned_state(ci.CC) == before


def test_shaping_and_servo_washout_do_not_clip_or_persist():
  ci, provider = settled(factor=43.)
  before = learned_state(ci.CC)
  provider.live_tuning = replace(provider.live_tuning, ecu_matched_long=True)
  for tick in range(100):
    step(ci, tick=tick)
  provider.live_tuning = replace(provider.live_tuning, ecu_matched_long=False)
  for tick in range(100, 100 + NIDEC_SERVO_LAG_FRAMES):
    step(ci, tick=tick)
  assert learned_state(ci.CC) == before
  assert not any("gas_factor" in entry for entry in provider.persisted)
  step(ci, tick=100 + NIDEC_SERVO_LAG_FRAMES)
  assert max(ci.CC.speed_factors.values()) == 16.
