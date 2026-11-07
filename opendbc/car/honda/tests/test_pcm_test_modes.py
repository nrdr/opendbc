"""Offline CAN and interface coverage for independent Civic PCM experiments."""
from dataclasses import replace
from unittest.mock import patch

import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC, HondaFlags
from opendbc.car.honda import mvl_longitudinal
from opendbc.car.honda.pcm_test_modes import FIXED_CRUISE_MPS
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from opendbc.sunnypilot.car.tests.runtime_config import make_test_car_config


def configured_experiment_params(fixed, legacy, *, car=CAR.HONDA_CIVIC, interceptor=False, stock=False,
                                 mvl=False, modified_eps=True, docs=False):
  config = make_test_car_config(honda_stock_longitudinal=stock)
  config = replace(config, honda=replace(config.honda, nidec_fixed_cruise=fixed, nidec_legacy_pcm=legacy,
                                       mvl_longitudinal=mvl, nidec_pcm_mode=2))
  fingerprint = {i: {} for i in range(7)}
  if interceptor:
    fingerprint[0][0x201] = 6
  firmware = [structs.CarParams.CarFw(ecu="eps", fwVersion=b"39990-TBA-A030,modified")] if modified_eps else []
  cp = CarInterface.get_params(car, fingerprint, firmware, alpha_long=True, is_release=False,
                               docs=docs, interface_config=config)
  return cp, fingerprint, firmware


@pytest.mark.parametrize("fixed,legacy", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("mvl", [False, True])
@pytest.mark.parametrize("modified_eps", [False, True])
def test_experiment_params_survive_modified_eps_admission(fixed, legacy, mvl, modified_eps):
  cp, fingerprint, firmware = configured_experiment_params(fixed, legacy, mvl=mvl, modified_eps=modified_eps)
  # This is the real intermediate state that previously discarded both requests.
  assert cp.dashcamOnly is modified_eps
  safety_before = [config.to_dict() for config in cp.safetyConfigs]
  sp = CarInterface.get_params_sp(cp, CAR.HONDA_CIVIC, fingerprint, firmware, alpha_long=True,
                                  is_release_sp=False, docs=False)
  assert not cp.dashcamOnly
  assert bool(sp.flags & HondaFlagsSP.EPS_MODIFIED) is modified_eps
  assert bool(cp.flags & HondaFlags.NIDEC_PCM_FIXED_CRUISE) is fixed
  assert bool(cp.flags & HondaFlags.NIDEC_PCM_LEGACY) is legacy
  assert bool(cp.flags & HondaFlags.MVL_LONGITUDINAL) is (mvl and not legacy)
  assert bool(cp.flags & HondaFlags.NIDEC_PCM_EXPERIMENTAL) is (not mvl and not legacy)
  assert not cp.flags & HondaFlags.NIDEC_PCM_ECU_MATCHED
  assert cp.openpilotLongitudinalControl and cp.pcmCruise and not sp.enableGasInterceptor
  assert sp.pcmCruiseSpeed is not fixed
  assert [config.to_dict() for config in cp.safetyConfigs] == safety_before
  assert sp.safetyParam == 0


@pytest.mark.parametrize("modified_eps", [False, True])
@pytest.mark.parametrize("configured", [False, True])
def test_experiment_params_default_off_after_eps_admission(modified_eps, configured):
  fingerprint = {i: {} for i in range(7)}
  firmware = [structs.CarParams.CarFw(ecu="eps", fwVersion=b"39990-TBA-A030,modified")] if modified_eps else []
  cp = CarInterface.get_params(CAR.HONDA_CIVIC, fingerprint, firmware, alpha_long=True, is_release=False,
                               docs=False, interface_config=make_test_car_config() if configured else None)
  sp = CarInterface.get_params_sp(cp, CAR.HONDA_CIVIC, fingerprint, firmware, alpha_long=True,
                                  is_release_sp=False, docs=False)
  assert not cp.dashcamOnly
  assert not cp.flags & (HondaFlags.NIDEC_PCM_FIXED_CRUISE | HondaFlags.NIDEC_PCM_LEGACY | HondaFlags.MVL_LONGITUDINAL)
  assert sp.pcmCruiseSpeed


@pytest.mark.parametrize("fixed,legacy", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("modified_eps", [False, True])
@pytest.mark.parametrize("car,interceptor,stock,docs", [
  (CAR.HONDA_CIVIC, True, False, False),
  (CAR.HONDA_CIVIC, False, True, False),
  (CAR.HONDA_CIVIC, False, False, True),
  (CAR.HONDA_CIVIC_BOSCH, False, False, False),
  (CAR.HONDA_CLARITY, False, False, False),
  (CAR.HONDA_ACCORD, False, False, False),
])
def test_experiment_params_reject_excluded_startups(fixed, legacy, modified_eps, car, interceptor, stock, docs):
  cp, fingerprint, firmware = configured_experiment_params(
    fixed, legacy, car=car, interceptor=interceptor, stock=stock, modified_eps=modified_eps, docs=docs, mvl=True,
  )
  sp = CarInterface.get_params_sp(cp, car, fingerprint, firmware, alpha_long=True, is_release_sp=False, docs=docs)
  assert not cp.flags & (HondaFlags.NIDEC_PCM_FIXED_CRUISE | HondaFlags.NIDEC_PCM_LEGACY)
  assert sp.pcmCruiseSpeed
  # A rejected legacy request must not displace the original MVL selection.
  assert bool(cp.flags & HondaFlags.MVL_LONGITUDINAL) is (not stock)


@pytest.mark.parametrize("fixed,legacy", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("field,value,modified_eps", [
  ("passive", True, True),
  ("dashcamOnly", True, False),  # No modified-EPS admission clears this final restriction.
  ("openpilotLongitudinalControl", False, True),
])
def test_experiment_params_final_gate_still_rejects_restricted_cp(fixed, legacy, field, value, modified_eps):
  cp, fingerprint, firmware = configured_experiment_params(fixed, legacy, modified_eps=modified_eps)
  assert bool(cp.flags & HondaFlags.NIDEC_PCM_FIXED_CRUISE) is fixed
  assert bool(cp.flags & HondaFlags.NIDEC_PCM_LEGACY) is legacy
  # Model a host restriction introduced between the two real parameter stages.
  setattr(cp, field, value)
  sp = CarInterface.get_params_sp(cp, CAR.HONDA_CIVIC, fingerprint, firmware, alpha_long=True,
                                  is_release_sp=False, docs=False)
  assert getattr(cp, field) is value
  assert not cp.flags & (HondaFlags.NIDEC_PCM_FIXED_CRUISE | HondaFlags.NIDEC_PCM_LEGACY)
  assert sp.pcmCruiseSpeed


def build(fixed=False, legacy=False, *, car=CAR.HONDA_CIVIC, interceptor=False, stock=False,
          wind=True, mvl=True, poison=False, modified_eps=False, pcm_mode=2, camera_messages=False,
          docs=False, passive=False, dashcam=False):
  config = make_test_car_config(honda_stock_longitudinal=stock)
  config = replace(config, honda=replace(config.honda, nidec_fixed_cruise=fixed, nidec_legacy_pcm=legacy,
                                       nidec_wind_brake=wind, nidec_pcm_mode=pcm_mode, mvl_longitudinal=mvl))
  provider = config.honda.provider
  provider.live_tuning = replace(provider.live_tuning, live_learning_gas=mvl)
  if poison:
    def reject_long(*args):
      raise AssertionError(f"legacy accessed longitudinal state: {args}")
    provider.initialize_live_learning_gas = reject_long
    provider.load_longitudinal_factors = reject_long
    provider.load_gas_alpha = reject_long

    def lateral_only(key, default):
      assert key.startswith("lat_accel_factor_"), key
      return default
    provider.load_mvl_param = lateral_only
  fp = {i: {} for i in range(7)}
  if interceptor:
    fp[0][0x201] = 6
  if camera_messages:
    fp[0][0x35E] = 8
  firmware = [structs.CarParams.CarFw(ecu="eps", fwVersion=b"39990-TBA-A030,modified")] if modified_eps else []
  cp = CarInterface.get_params(car, fp, firmware, alpha_long=True, is_release=False, docs=docs, interface_config=config)
  if modified_eps:
    assert cp.dashcamOnly  # Exercise the temporary restriction before SP admission.
  cp.passive = passive
  if dashcam:
    cp.dashcamOnly = True
  sp = CarInterface.get_params_sp(cp, car, fp, firmware, alpha_long=True, is_release_sp=False, docs=docs)
  ci = CarInterface(cp, sp, config)
  ci.update([])
  ci.CS.out.vEgo = 20.0
  ci.CS.out.aEgo = 0.0
  ci.CS.out.canValid = True
  ci.CS.out.cruiseState.speed = 20.0
  ci.CS.out.cruiseState.available = True
  ci.CS.out.cruiseState.enabled = True
  ci.CS.out.standstill = False
  ci.CS.steer_control_active = True
  return ci, provider


def control(accel=0.3, brake=0.0, speed=21.0):
  cc = structs.CarControl(enabled=True, longActive=True)
  cc.actuators.accel = accel
  cc.actuators.brake = brake
  cc.actuators.speed = speed
  cc.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
  cc.hudControl.speedVisible = True
  cc.hudControl.setSpeed = 20.0
  return cc


def update(ci, cc, ticks=1):
  messages = [("ACC_HUD", 10)] + ([] if ci.CP.flags & HondaFlags.BOSCH else [("BRAKE_COMMAND", 50)])
  parser = CANParser(DBC[ci.CP.carFingerprint][Bus.pt], messages, ci.CC.CAN.pt)
  frames = []
  for _ in range(ticks):
    _, sends = ci.CC.update(cc.as_reader(), structs.CarControlSP(), ci.CS, ci.CC.frame * 10_000_000)
    parser.update([(ci.CC.frame * 10_000_000, sends)])
    frames.extend(sends)
  return parser.vl, frames


@pytest.mark.parametrize("fixed,legacy", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("mvl", [False, True])
@pytest.mark.parametrize("pcm_mode", [0, 1, 2])
def test_modified_eps_experiments_reach_controller_and_packed_can(fixed, legacy, mvl, pcm_mode):
  ci, _ = build(fixed, legacy, modified_eps=True, camera_messages=True, mvl=mvl, pcm_mode=pcm_mode, poison=legacy)
  baseline, _ = build(modified_eps=True, camera_messages=True, mvl=mvl, pcm_mode=pcm_mode)
  assert not ci.CP.dashcamOnly
  assert ci.CP_SP.flags == baseline.CP_SP.flags == int(HondaFlagsSP.EPS_MODIFIED | HondaFlagsSP.HAS_CAMERA_MESSAGES)
  assert ci.CC.fixed_cruise is fixed and ci.CC.legacy_pcm is legacy
  assert ci.CC.mvl_longitudinal is (mvl and not legacy)
  assert ci.CC.civic_pcm is (pcm_mode == 2 and not mvl and not legacy)
  assert ci.CC.ecu_matched_pcm is (pcm_mode == 1 and not mvl and not legacy)
  assert ci.CP.pcmCruise and ci.CP_SP.pcmCruiseSpeed is not fixed
  assert [x.to_dict() for x in ci.CP.safetyConfigs] == [x.to_dict() for x in baseline.CP.safetyConfigs]
  assert ci.CP_SP.safetyParam == baseline.CP_SP.safetyParam
  signals, frames = update(ci, control(), 20)
  baseline_signals, baseline_frames = update(baseline, control(), 20)
  assert frames and baseline_frames
  assert signals["ACC_HUD"]["CRUISE_SPEED"] == (145 if fixed else 72)
  assert baseline_signals["ACC_HUD"]["CRUISE_SPEED"] == 72
  assert 0 <= signals["ACC_HUD"]["PCM_GAS"] <= 198
  assert 0 <= signals["BRAKE_COMMAND"]["COMPUTER_BRAKE"] <= 255
  if not fixed and not legacy:
    assert frames == baseline_frames


def test_modified_eps_fixed_only_live_configuration_reaches_packed_can():
  # Saved Fixed=1, Legacy=0, MVL=0, PCM mode=0, with EPS + camera detection.
  baseline, _ = build(modified_eps=True, camera_messages=True, mvl=False, pcm_mode=0)
  fixed, _ = build(True, False, modified_eps=True, camera_messages=True, mvl=False, pcm_mode=0)
  assert baseline.CP.flags == 272
  assert fixed.CP.flags == 272 | HondaFlags.NIDEC_PCM_FIXED_CRUISE.value
  assert fixed.CP_SP.flags == baseline.CP_SP.flags == 10
  assert baseline.CP_SP.pcmCruiseSpeed and not fixed.CP_SP.pcmCruiseSpeed
  assert fixed.CC.fixed_cruise and not fixed.CC.legacy_pcm and not fixed.CC.mvl_longitudinal
  assert not fixed.CC.civic_pcm and not fixed.CC.ecu_matched_pcm
  assert [x.to_dict() for x in fixed.CP.safetyConfigs] == [x.to_dict() for x in baseline.CP.safetyConfigs]
  assert fixed.CP_SP.safetyParam == baseline.CP_SP.safetyParam
  assert CarInterface.get_pid_accel_limits(fixed.CP, fixed.CP_SP, 20., 20.) == (-4., 2.)
  assert CarInterface.get_pid_accel_limits(baseline.CP, baseline.CP_SP, 20., 20.) == (-4., .2)
  assert update(baseline, control(), 20)[0]["ACC_HUD"]["CRUISE_SPEED"] == 72
  assert update(fixed, control(), 20)[0]["ACC_HUD"]["CRUISE_SPEED"] == 145


@pytest.mark.parametrize("mvl", [False, True])
@pytest.mark.parametrize("pcm_mode", [0, 1, 2])
@pytest.mark.parametrize("options", [
  pytest.param({"interceptor": True}, id="interceptor"),
  pytest.param({"stock": True}, id="stock-longitudinal"),
  pytest.param({"car": CAR.HONDA_CIVIC_BOSCH}, id="bosch-civic"),
  pytest.param({"car": CAR.HONDA_CLARITY}, id="non-civic-nidec"),
  pytest.param({"car": CAR.HONDA_ACCORD}, id="non-civic-bosch"),
  pytest.param({"docs": True}, id="docs"),
  pytest.param({"passive": True}, id="passive"),
  pytest.param({"dashcam": True, "modified_eps": False}, id="final-dashcam"),
])
def test_modified_eps_excluded_experiments_preserve_mode_safety_and_packets(mvl, pcm_mode, options):
  kwargs = {"modified_eps": True, "camera_messages": True, "mvl": mvl, "pcm_mode": pcm_mode, **options}
  requested, _ = build(True, True, **kwargs)
  baseline, _ = build(**kwargs)
  assert not requested.CC.fixed_cruise and not requested.CC.legacy_pcm
  for mode in ("mvl_longitudinal", "civic_pcm", "ecu_matched_pcm"):
    assert getattr(requested.CC, mode) == getattr(baseline.CC, mode)
  assert requested.CP.to_dict() == baseline.CP.to_dict()  # Includes all safety configs and mode flags.
  assert requested.CP_SP == baseline.CP_SP  # Includes SP safetyParam and PCM cruise ownership.
  assert update(requested, control(), 50)[1] == update(baseline, control(), 50)[1]


@pytest.mark.parametrize("fixed,legacy", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_modes_and_preserved_safety_configuration(fixed, legacy):
  ci, _ = build(fixed, legacy, poison=legacy)
  baseline, _ = build()
  assert ci.CC.fixed_cruise is fixed
  assert ci.CC.legacy_pcm is legacy
  assert ci.CC.mvl_longitudinal is not legacy
  assert ci.CP.pcmCruise is True
  assert ci.CP_SP.pcmCruiseSpeed is not fixed
  assert [x.to_dict() for x in ci.CP.safetyConfigs] == [x.to_dict() for x in baseline.CP.safetyConfigs]
  signals, _ = update(ci, control(), 20)
  assert signals["ACC_HUD"]["CRUISE_SPEED"] == (round(FIXED_CRUISE_MPS * 3.6) if fixed else 72)
  assert signals["ACC_HUD"]["PCM_SPEED"] < FIXED_CRUISE_MPS * 3.6
  if legacy:
    assert not ci.CC.nidec_wind_brake
    assert not ci.CC.civic_pcm and not ci.CC.ecu_matched_pcm
    assert not hasattr(ci.CC, "gas_factors")


@pytest.mark.parametrize("car,interceptor,stock", [
  (CAR.HONDA_CIVIC, True, False), (CAR.HONDA_CIVIC, False, True),
  (CAR.HONDA_CIVIC_BOSCH, False, False), (CAR.HONDA_CLARITY, False, False),
  (CAR.HONDA_ACCORD, False, False),
])
def test_excluded_hardware_output_unchanged(car, interceptor, stock):
  a, _ = build(True, True, car=car, interceptor=interceptor, stock=stock)
  b, _ = build(car=car, interceptor=interceptor, stock=stock)
  assert not a.CC.fixed_cruise and not a.CC.legacy_pcm
  assert update(a, control(), 50)[1] == update(b, control(), 50)[1]


def test_legacy_does_not_call_mvl_or_apply_wind_or_saved_longitudinal_state():
  a, pa = build(True, True, wind=True, poison=True)
  b, pb = build(True, True, wind=False, poison=True)
  with patch.object(mvl_longitudinal, "prepare", side_effect=AssertionError("MVL prepare")), \
       patch.object(mvl_longitudinal, "commands", side_effect=AssertionError("MVL commands")), \
       patch.object(mvl_longitudinal, "persist", side_effect=AssertionError("MVL persist")):
    for tick in range(1000):
      brake = .4 if 100 < tick < 600 else 0.0
      cc = control(-1.0 if brake else .3, brake)
      cc.orientationNED = [0., .2 if tick % 2 else -.2, 0.]
      assert update(a, cc)[1] == update(b, cc)[1]
  assert all(all(k.startswith("lat_accel_factor_") for k in row) for row in pa.persisted + pb.persisted)


@pytest.mark.parametrize("guard", ["disabled", "inactive", "cancel", "gas", "brake", "invalid_can", "acc_fault", "off_state"])
def test_legacy_cancellation_and_driver_override(guard):
  ci, _ = build(True, True)
  cc = control()
  update(ci, cc, 20)
  if guard == "disabled":
    cc.enabled = False
  elif guard == "inactive":
    cc.longActive = False
  elif guard == "cancel":
    cc.cruiseControl.cancel = True
  elif guard == "gas":
    ci.CS.out.gasPressed = True
  elif guard == "brake":
    ci.CS.out.brakePressed = True
  elif guard == "invalid_can":
    ci.CS.out.canValid = False
  elif guard == "acc_fault":
    ci.CS.out.accFaulted = True
  else:
    cc.actuators.longControlState = structs.CarControl.Actuators.LongControlState.off
  signals, _ = update(ci, cc, 20)
  assert signals["ACC_HUD"]["PCM_GAS"] == 0
  assert signals["ACC_HUD"]["PCM_SPEED"] == 0
  assert signals["BRAKE_COMMAND"]["COMPUTER_BRAKE"] == 0
  if guard not in ("gas", "off_state"):
    assert signals["ACC_HUD"]["CRUISE_SPEED"] != round(FIXED_CRUISE_MPS * 3.6)


def test_legacy_friction_brakes_and_stock_aeb_keep_priority():
  ci, _ = build(True, True)
  ci.CS.out.stockAeb = True
  signals, _ = update(ci, control(-2.0, .5, 18.0), 100)
  assert signals["BRAKE_COMMAND"]["COMPUTER_BRAKE"] > 0
  assert signals["ACC_HUD"]["PCM_GAS"] == signals["ACC_HUD"]["PCM_SPEED"] == 0
  assert signals["BRAKE_COMMAND"]["AEB_REQ_1"] == 0  # Panda arbitrates stock camera braking separately.


@pytest.mark.parametrize("field", ["speed", "accel", "brake"])
def test_invalid_legacy_reference_cannot_command_propulsion(field):
  ci, _ = build(True, True)
  cc = control()
  setattr(cc.actuators, field, float("nan"))
  signals, _ = update(ci, cc, 20)
  assert signals["ACC_HUD"]["PCM_GAS"] == signals["ACC_HUD"]["PCM_SPEED"] == 0


def test_fixed_mode_removes_only_near_user_set_speed_limit():
  a, _ = build(True)
  b, _ = build()
  fixed_limits = CarInterface.get_pid_accel_limits(a.CP, a.CP_SP, 20., 20.)
  normal_limits = CarInterface.get_pid_accel_limits(b.CP, b.CP_SP, 20., 20.)
  assert fixed_limits == (-4.0, 2.0)
  assert normal_limits[1] == .2


@pytest.mark.parametrize("speed_mode", [0, 1, 2, 3])
@pytest.mark.parametrize("distance_mode", [0, 1, 2, 3])
def test_fixed_ceiling_is_not_overwritten_by_dashboard_preferences(speed_mode, distance_mode):
  ci, provider = build(True, True)
  provider.live_tuning = replace(provider.live_tuning, alt_dashboard_speed=speed_mode, alt_dashboard_distance=distance_mode,
                                 sub_mode_enabled=True, sub_mode_until=10000.)
  signals, _ = update(ci, control(), 20)
  assert signals["ACC_HUD"]["CRUISE_SPEED"] == 145
  assert signals["ACC_HUD"]["ACC_ON"] == 1


@pytest.mark.parametrize("fixed", [False, True])
def test_legacy_full_stop_holds_brake_without_propulsion(fixed):
  ci, _ = build(fixed, True)
  ci.CS.out.vEgo = 0.
  ci.CS.out.standstill = True
  cc = control(-.5, .25, 0.)
  cc.actuators.longControlState = structs.CarControl.Actuators.LongControlState.stopping
  signals, _ = update(ci, cc, 300)
  assert signals["BRAKE_COMMAND"]["COMPUTER_BRAKE"] > 0
  assert signals["ACC_HUD"]["PCM_GAS"] == signals["ACC_HUD"]["PCM_SPEED"] == 0


@pytest.mark.parametrize("active,aeb,cancel,gas", [(True, False, False, False), (True, False, False, True),
                                                 (False, True, False, False), (True, False, True, False),
                                                 (False, False, True, False)])
def test_legacy_retains_honda_cruise_override_semantics(active, aeb, cancel, gas):
  ci, _ = build(True, True)
  cc = control()
  cc.longActive, cc.cruiseControl.cancel = active, cancel
  ci.CS.out.stockAeb, ci.CS.out.gasPressed = aeb, gas
  signals, _ = update(ci, cc, 20)
  assert signals["BRAKE_COMMAND"]["CRUISE_OVERRIDE"] == int(active or aeb)
  assert signals["BRAKE_COMMAND"]["CRUISE_CANCEL_CMD"] == int(cancel)
