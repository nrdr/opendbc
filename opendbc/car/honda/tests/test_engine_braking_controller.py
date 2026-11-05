"""Wire-level isolation checks, not a model of Honda's engine response."""
import pytest

from opendbc.car import structs
from opendbc.car.honda.tests.test_civic_pcm import interface, control
from opendbc.car.honda.values import CAR


def drive(ci, tick, *, permitted=True, request=True, accel=.2):
  stamp = (tick + 100) * 10_000_000
  cc_sp = structs.CarControlSP(pcmEngineBrakeAllowed=permitted, pcmEngineBrakeRequest=request, pcmEngineBrakeMonoTime=stamp)
  return ci.apply(control(accel).as_reader(), cc_sp, stamp)


@pytest.mark.parametrize('mode', [0, 1, 2])
def test_default_off_never_changes_wire(mode):
  left, _ = interface(pcm_mode=mode)
  right, _ = interface(pcm_mode=mode)
  for tick in range(400):
    now = (tick + 100) * 10_000_000
    _, expected = left.apply(control(.2).as_reader(), structs.CarControlSP(), now)
    _, actual = drive(right, tick, permitted=False)
    assert actual == expected


@pytest.mark.parametrize('car,interceptor', [(CAR.HONDA_CLARITY, False), (CAR.HONDA_CLARITY, True),
  (CAR.HONDA_CIVIC, True), (CAR.HONDA_CIVIC_BOSCH, False), (CAR.HONDA_RIDGELINE, False)])
def test_other_paths_ignore_even_a_spurious_permission(car, interceptor):
  left, _ = interface(car, interceptor=interceptor, pcm_mode=0)
  right, _ = interface(car, interceptor=interceptor, pcm_mode=0)
  left.CS.out.canValid = right.CS.out.canValid = True
  for tick in range(400):
    _, expected = drive(left, tick, permitted=False)
    _, actual = drive(right, tick)
    assert actual == expected


def test_fade_reaches_zero_and_freezes_learning():
  ci, config = interface(pcm_mode=0, learning=True)
  ci.CS.out.canValid = True
  from opendbc.car.honda.tests.test_mvl_nrdr_integration import learned_state
  before = learned_state(ci.CC)
  for tick in range(310):
    drive(ci, tick)
  assert ci.CC.engine_brake_fade.weight == pytest.approx(1)
  assert ci.CC.pcm_gas_sent == 0
  assert ci.CC.pcm_speed_sent == pytest.approx(0, abs=.003)
  assert learned_state(ci.CC) == before
  assert all(all(key.startswith('lat_accel_factor_') for key in item) for item in config.honda.provider.persisted)


@pytest.mark.parametrize('override', ['brake', 'gas', 'lead', 'aeb', 'invalid_can', 'cancel', 'stop'])
def test_priority_overrides_clear_fade(override):
  ci, _ = interface(pcm_mode=0)
  ci.CS.out.canValid = True
  for tick in range(100):
    drive(ci, tick)
  assert ci.CC.engine_brake_fade.weight > 0
  cc = control(.2)
  sp = structs.CarControlSP(pcmEngineBrakeAllowed=True, pcmEngineBrakeRequest=True, pcmEngineBrakeMonoTime=2_000_000_000)
  if override == 'brake':
    cc.actuators.accel = -2.
  elif override == 'gas':
    ci.CS.out.gasPressed = True
    cc.longActive = False
  elif override == 'lead':
    sp.leadOne.status = True
  elif override == 'aeb':
    ci.CS.out.stockAeb = True
  elif override == 'invalid_can':
    ci.CS.out.canValid = False
  elif override == 'cancel':
    cc.cruiseControl.cancel = True
  else:
    cc.actuators.longControlState = structs.CarControl.Actuators.LongControlState.stopping
  ci.apply(cc.as_reader(), sp, 2_000_000_000)
  assert ci.CC.engine_brake_fade.weight == 0
