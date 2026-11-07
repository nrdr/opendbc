"""Pure command-law tests; no vehicle, CAN send, Params, or learner I/O."""
import math

import pytest

from opendbc.car.honda.legacy_pcm import (
  LegacyPCMCommand, LegacyPCMController, brake_hysteresis, brake_limit, brake_pump_hysteresis,
  calc_accel_override, compute_gb,
)


def command(controller, **overrides):
  args = dict(active=True, v_ego=20.0, a_ego=0.0, v_target=20.0, a_target=0.0, brake=0.0)
  args.update(overrides)
  return controller.update(**args)


@pytest.mark.parametrize('speed,expected', [(0.0, -.15), (1.15, -.075), (2.3, 0.), (35., 0.)])
def test_fixed_creep_no_wind(speed, expected):
  assert compute_gb(0.0, speed) == pytest.approx(expected)
  assert compute_gb(4.8, speed) == pytest.approx(1.0 + expected)


@pytest.mark.parametrize('speed,expected', [(0., 1.), (5., 1.), (12.5, .9), (20., .8), (40., .8)])
def test_historical_brake_limit(speed, expected):
  assert brake_limit(speed) == pytest.approx(expected)


@pytest.mark.parametrize('a_ego,a_target,v_ego,v_target,expected', [
  (0., 0., 10., 10., .714), (0., 0., 20., 20., 1.), (0., 0., 20., 19.5, .1),
  (1.1, 0., 20., 20., .1), (1.1, 0., 20., 21., 1.),
  (0., 2.4, 10., 10., 1.5), (0., 2.4, 10., 9.5, .15),
])
def test_exact_old_accel_override(a_ego, a_target, v_ego, v_target, expected):
  assert calc_accel_override(a_ego, a_target, v_ego, v_target) == pytest.approx(expected)


def test_allowance_uses_user_reference_not_fixed_car_governor():
  controller = LegacyPCMController()
  assert command(controller, v_target=19.5).pcm_accel == 19
  assert command(controller, v_target=20.).pcm_accel == 198
  # The helper has no 90-mph governor input: independent toggle cannot alter it.
  assert command(controller, v_target=19.5).pcm_speed == 19.5


def test_brake_hysteresis():
  assert brake_hysteresis(.019, False, 0.) == (0., False, 0.)
  assert brake_hysteresis(.02, False, 0.) == (.01, True, .01)
  assert brake_hysteresis(.025, True, .02) == (.02, True, .02)
  assert brake_hysteresis(.004, True, .02) == (0., False, 0.)


def test_exact_pump_increment_hold_decrease():
  assert brake_pump_hysteresis(10, 0, False)
  assert brake_pump_hysteresis(10, 10, True)
  assert not brake_pump_hysteresis(9, 10, True)
  assert not brake_pump_hysteresis(9, 9, False)


def test_100hz_brake_shape_50hz_sent_pump_history_and_exclusion():
  controller = LegacyPCMController()
  first = command(controller, brake=.5, send_brake=True)
  assert first.brake_fraction == pytest.approx(.01)
  assert first.apply_brake == 2 and first.pump_on
  odd = command(controller, brake=.5, send_brake=False)
  assert odd.brake_fraction == pytest.approx(.02)
  assert controller.apply_brake_last == 2
  even = command(controller, brake=.5, send_brake=True)
  assert even.apply_brake == 7 and controller.apply_brake_last == 7
  assert first.pcm_accel == odd.pcm_accel == even.pcm_accel == 0
  assert first.pcm_speed == odd.pcm_speed == even.pcm_speed == 0.


def test_global_caps_and_existing_brake_release_limit():
  controller = LegacyPCMController()
  for frame in range(110):
    result = command(controller, brake=1., v_ego=0., a_target=-10., send_brake=frame % 2 == 0)
    assert 0 <= result.apply_brake <= 255
    assert 0 <= result.pcm_accel <= 198
  previous = controller.apply_brake_last
  result = command(controller, brake=0.)
  assert result.apply_brake == previous - 32
  assert not result.pump_on
  assert result.pcm_accel == result.pcm_speed == 0
  # A reduced caller cap is still authoritative, including during release.
  assert command(controller, brake=1., brake_max=128).apply_brake <= 127
  controller.reset()
  assert command(controller, a_target=10., gas_max=1000).pcm_accel == 198
  assert command(controller, a_target=10., gas_max=100).pcm_accel == 100


@pytest.mark.parametrize('override', [dict(active=False), dict(cancel=True), dict(gas_pressed=True),
                                      dict(brake_pressed=True), dict(v_target=math.nan), dict(a_target=math.inf)])
def test_override_or_invalid_input_resets(override):
  controller = LegacyPCMController()
  command(controller, brake=.5)
  assert command(controller, **override) == LegacyPCMCommand()
  assert controller.apply_brake_last == 0 and not controller.pump_on


def test_aeb_removes_propulsion_without_stealing_service_brakes():
  controller = LegacyPCMController()
  result = command(controller, stock_aeb=True)
  assert result.pcm_accel == result.pcm_speed == 0
  result = command(controller, stock_aeb=True, brake=.5)
  assert result.apply_brake > 0 and result.pump_on


def test_brake_discount_uses_normalized_demand():
  result = command(LegacyPCMController(), brake=.01, speed_offset=-.2)
  assert result.apply_brake == 0
  assert result.pcm_speed == pytest.approx(19.8 * .97)
