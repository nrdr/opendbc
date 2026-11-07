"""Real Honda interface/controller/packed-CAN coverage; no vehicle connection."""
import pytest

from opendbc.can import CANParser, CANPacker
from opendbc.car import Bus, structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda import hondacan
from opendbc.car.honda.values import CAR, DBC, CruiseButtons, HondaFlags
from opendbc.sunnypilot.car.honda.icbm import IntelligentCruiseButtonManagementInterface
from opendbc.sunnypilot.car.honda.icbm_eligibility import civic_icbm_available
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from opendbc.sunnypilot.car.interfaces import setup_interfaces
from opendbc.sunnypilot.car.tests.runtime_config import make_test_car_config


def build(car=CAR.HONDA_CIVIC, *, stock=True, interceptor=False, modified_eps=False):
  config = make_test_car_config(honda_stock_longitudinal=stock)
  fp = {i: {} for i in range(7)}
  if interceptor:
    fp[0][0x201] = 6
  fw = [structs.CarParams.CarFw(ecu="eps", fwVersion=b"39990-TBA-A030,modified")] if modified_eps else []
  cp = CarInterface.get_params(car, fp, fw, alpha_long=not stock, is_release=False, docs=False, interface_config=config)
  sp = CarInterface.get_params_sp(cp, car, fp, fw, alpha_long=not stock, is_release_sp=False, docs=False)
  setup_interfaces(CarInterface, cp, sp, config)
  ci = CarInterface(cp, sp, config)
  ci.update([])
  ci.CS.out.vEgo = 20.0
  ci.CS.out.canValid = True
  ci.CS.out.cruiseState.enabled = True
  ci.CS.out.cruiseState.available = True
  ci.CS.out.cruiseState.speed = 20.0
  ci.CS.out.standstill = False
  ci.CS.cruise_buttons = ci.CS.cruise_setting = 0
  ci.CS.steer_control_active = True
  return ci


def controls(direction="increase"):
  cc = structs.CarControl(enabled=True)
  cc.hudControl.setSpeed = 20.0
  sp = structs.CarControlSP()
  sp.intelligentCruiseButtonManagement.sendButton = direction
  return cc, sp


def decode(cp, msg):
  parser = CANParser(DBC[cp.carFingerprint][Bus.pt], [("SCM_BUTTONS", 25)], msg[2])
  parser.update([(1_000_000_000, [msg])])
  return dict(parser.vl["SCM_BUTTONS"])


def adapter(ci, cc, sp, frame=6):
  return IntelligentCruiseButtonManagementInterface.update(
    ci.CC, cc, sp, ci.CS, ci.CC.packer, frame, ci.CC.last_button_frame, ci.CC.CAN,
  )


@pytest.mark.parametrize("stock", [False, True])
@pytest.mark.parametrize("modified_eps", [False, True])
@pytest.mark.parametrize("interceptor", [False, True])
def test_hardware_capability_is_independent_of_selection(stock, modified_eps, interceptor):
  ci = build(stock=stock, interceptor=interceptor, modified_eps=modified_eps)
  assert civic_icbm_available(ci.CP, ci.CP_SP) is (not interceptor)
  assert ci.CP_SP.intelligentCruiseButtonManagementAvailable is (not interceptor)
  assert ci.CP_SP.pcmCruiseSpeed  # This helper does not enable ICBM or alter preferences.
  assert ci.CP.openpilotLongitudinalControl is (not stock)
  assert bool(ci.CP_SP.flags & HondaFlagsSP.STOCK_LONGITUDINAL) is stock


@pytest.mark.parametrize("car", [CAR.HONDA_CRV, CAR.HONDA_CLARITY, CAR.HONDA_CIVIC_BOSCH])
def test_other_hardware_does_not_use_civic_eligibility(car):
  ci = build(car)
  assert not civic_icbm_available(ci.CP, ci.CP_SP)
  if not ci.CP.flags & HondaFlags.BOSCH:
    assert not ci.CP_SP.intelligentCruiseButtonManagementAvailable


@pytest.mark.parametrize("direction,button", [("increase", CruiseButtons.RES_ACCEL), ("decrease", CruiseButtons.DECEL_SET)])
@pytest.mark.parametrize("long_active", [False, True])
def test_full_controller_sends_only_buttons_with_inherited_cadence_and_counters(direction, button, long_active):
  ci = build()
  ci.CP_SP.pcmCruiseSpeed = False
  cc, sp = controls(direction)
  cc.longActive = long_active
  packets = []
  for frame in range(31):
    _, sends = ci.CC.update(cc.as_reader(), sp, ci.CS, frame * 10_000_000)
    assert not {0x30C, 0x1FA, 0x200} & {msg[0] for msg in sends}
    for msg in sends:
      if msg[0] == 0x296:
        assert msg[2] == ci.CC.CAN.camera and len(msg[1]) == 4
        packets.append((frame, decode(ci.CP, msg)))
  assert [frame for frame, _ in packets] == [6, 12, 18, 24, 30]
  assert [v["COUNTER"] for _, v in packets] == [0, 1, 2, 3, 0]
  assert all(v["CRUISE_BUTTONS"] == button and v["CRUISE_SETTING"] == 0 for _, v in packets)


@pytest.mark.parametrize("owner,field,value", [
  ("cp", "openpilotLongitudinalControl", True), ("cp", "passive", True), ("cp", "dashcamOnly", True),
  ("sp", "pcmCruiseSpeed", True), ("sp", "enableGasInterceptor", True), ("sp", "flags", 0),
  ("cc", "enabled", False),
  ("cruise_control", "cancel", True), ("cruise_control", "resume", True), ("cruise_control", "override", True),
  ("cruise_state", "enabled", False), ("cruise_state", "standstill", True),
  ("out", "standstill", True), ("out", "gasPressed", True), ("out", "brakePressed", True),
  ("cs", "cruise_buttons", CruiseButtons.RES_ACCEL), ("cs", "cruise_buttons", CruiseButtons.DECEL_SET),
  ("cs", "cruise_buttons", CruiseButtons.CANCEL), ("cs", "cruise_buttons", CruiseButtons.MAIN),
  ("cs", "cruise_setting", 1), ("cs", "cruise_setting", 3),
])
def test_adapter_rechecks_runtime_guards_even_with_stale_send_button(owner, field, value):
  ci = build()
  ci.CP_SP.pcmCruiseSpeed = False
  cc, sp = controls()
  targets = {"cp": ci.CP, "sp": ci.CP_SP, "cc": cc, "cs": ci.CS, "out": ci.CS.out,
             "cruise_control": cc.cruiseControl, "cruise_state": ci.CS.out.cruiseState}
  setattr(targets[owner], field, value)
  assert adapter(ci, cc, sp) == []
  assert ci.CC.last_button_frame == 0


@pytest.mark.parametrize("car", [CAR.HONDA_CRV, CAR.HONDA_CLARITY])
def test_other_nidecs_cannot_emit_stale_icbm(car):
  ci = build(car)
  ci.CP_SP.pcmCruiseSpeed = False
  cc, sp = controls()
  assert adapter(ci, cc, sp) == []


@pytest.mark.parametrize("field,button", [("cancel", CruiseButtons.CANCEL), ("resume", CruiseButtons.RES_ACCEL)])
def test_stock_cancel_resume_route_takes_priority_over_stale_icbm(field, button):
  ci = build()
  ci.CP_SP.pcmCruiseSpeed = False
  cc, sp = controls("decrease")
  setattr(cc.cruiseControl, field, True)
  cc.enabled = field != "cancel"
  ci.CC.frame = 6
  _, sends = ci.CC.update(cc.as_reader(), sp, ci.CS, 60_000_000)
  buttons = [msg for msg in sends if msg[0] == 0x296]
  assert len(buttons) == 1
  assert buttons[0][2] == ci.CC.CAN.camera
  assert decode(ci.CP, buttons[0])["CRUISE_BUTTONS"] == button


@pytest.mark.parametrize("car", [CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CIVIC_2022])
def test_bosch_route_is_preserved_but_op_long_rejects_stale_icbm(car):
  ci = build(car)
  cc, sp = controls()
  packets = adapter(ci, cc, sp)
  assert len(packets) == 1
  expected_bus = ci.CC.CAN.camera if ci.CP.flags & HondaFlags.BOSCH_RADARLESS else ci.CC.CAN.pt
  assert packets[0][2] == expected_bus
  assert decode(ci.CP, packets[0])["CRUISE_BUTTONS"] == CruiseButtons.RES_ACCEL
  ci.CP.openpilotLongitudinalControl = True
  assert adapter(ci, cc, sp, frame=12) == []


def test_civic_helper_pack_does_not_log_unknown_ambient_signal(monkeypatch):
  from opendbc.car import carlog
  ci = build()
  ci.CP_SP.pcmCruiseSpeed = False
  cc, sp = controls()

  def fail(*args, **kwargs):
    pytest.fail(f"packer emitted an error: {args}")

  monkeypatch.setattr(carlog.carlog, "error", fail)
  ci.CC.packer = CANPacker(DBC[ci.CP.carFingerprint][Bus.pt])
  assert len(adapter(ci, cc, sp)) == 1


@pytest.mark.parametrize("car", [CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CIVIC_2022, CAR.HONDA_CRV, CAR.HONDA_CLARITY])
def test_non_civic_button_payload_and_default_bus_are_unchanged(car):
  ci = build(car)
  dbc = DBC[car][Bus.pt]
  bus = ci.CC.CAN.camera if ci.CP.flags & HondaFlags.BOSCH_RADARLESS else ci.CC.CAN.pt
  expected = CANPacker(dbc).make_can_msg("SCM_BUTTONS", bus, {
    "CRUISE_BUTTONS": CruiseButtons.RES_ACCEL, "CRUISE_SETTING": 0, "AMBIENT_LIGHT_MAYBE": 73,
  })
  actual = hondacan.spam_buttons_command(CANPacker(dbc), ci.CC.CAN, CruiseButtons.RES_ACCEL, 0, 73, ci.CP)
  assert actual == expected
