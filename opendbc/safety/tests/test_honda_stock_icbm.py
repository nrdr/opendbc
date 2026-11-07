"""Native compiled safety hooks exercised with real Civic packed CAN messages."""
import sys

import pytest

# The upstream libsafety builder requires a POSIX cc/shared-library environment.
# Do not substitute a Python/AST model for the actual C safety implementation.
if sys.platform == "win32":
  pytest.skip("native libsafety requires the Linux test overlay", allow_module_level=True)

from opendbc.can import CANPacker
from opendbc.car import structs
from opendbc.car.honda.values import CruiseButtons
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.sunnypilot.car.honda.values_ext import HondaSafetyFlagsSP


def packet(packer, name, bus, values):
  addr, data, bus = packer.make_can_msg(name, bus, values)
  return libsafety_py.make_CANPacket(addr, bus, data)


@pytest.fixture
def native():
  safety = libsafety_py.libsafety
  safety.set_current_safety_param_sp(HondaSafetyFlagsSP.STOCK_LONGITUDINAL)
  assert safety.set_safety_hooks(structs.CarParams.SafetyModel.hondaNidec, 0) == 0
  safety.init_tests()
  packer = CANPacker("honda_civic_touring_2016_can_generated")
  assert safety.safety_rx_hook(packet(packer, "SCM_FEEDBACK", 0, {"MAIN_ON": 1}))
  return safety, packer


def cruise(native, enabled, **values):
  safety, packer = native
  assert safety.safety_rx_hook(packet(packer, "POWERTRAIN_DATA", 0, {"ACC_STATUS": enabled, **values}))


def button(native, value, *, bus=2, setting=0):
  _, packer = native
  return packet(packer, "SCM_BUTTONS", bus, {"CRUISE_BUTTONS": value, "CRUISE_SETTING": setting})


@pytest.mark.parametrize("engaged", [False, True])
@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize("value", range(8))
def test_camera_button_authority_matrix(native, engaged, allowed, value):
  safety, _ = native
  cruise(native, engaged)
  safety.set_controls_allowed(allowed)
  expected = value == CruiseButtons.CANCEL or (value in (CruiseButtons.RES_ACCEL, CruiseButtons.DECEL_SET) and engaged and allowed)
  assert bool(safety.safety_tx_hook(button(native, value))) is expected


@pytest.mark.parametrize("value", [CruiseButtons.RES_ACCEL, CruiseButtons.DECEL_SET])
def test_latched_nidec_permission_is_not_stock_cruise_engagement(native, value):
  safety, _ = native
  cruise(native, True)
  assert safety.get_controls_allowed()
  assert safety.safety_tx_hook(button(native, value))
  cruise(native, False)
  assert safety.get_controls_allowed()  # Preserve the existing Nidec stop/braking state machine.
  assert not safety.safety_tx_hook(button(native, value))
  assert safety.safety_tx_hook(button(native, CruiseButtons.CANCEL))


@pytest.mark.parametrize("value", [CruiseButtons.RES_ACCEL, CruiseButtons.DECEL_SET])
@pytest.mark.parametrize("pedal", ["PEDAL_GAS", "BRAKE_PRESSED"])
def test_driver_pedals_reject_adjustment_even_if_permission_is_set(native, value, pedal):
  safety, _ = native
  cruise(native, True, **{pedal: 1})
  safety.set_controls_allowed(True)
  assert not safety.safety_tx_hook(button(native, value))
  assert safety.safety_tx_hook(button(native, CruiseButtons.CANCEL))


@pytest.mark.parametrize("sp_flags", [0, HondaSafetyFlagsSP.GAS_INTERCEPTOR,
                                     HondaSafetyFlagsSP.STOCK_LONGITUDINAL | HondaSafetyFlagsSP.GAS_INTERCEPTOR,
                                     HondaSafetyFlagsSP.STOCK_LONGITUDINAL | HondaSafetyFlagsSP.NIDEC_HYBRID])
@pytest.mark.parametrize("value", [CruiseButtons.RES_ACCEL, CruiseButtons.DECEL_SET, CruiseButtons.CANCEL])
def test_camera_permission_does_not_leak_to_other_nidec_configurations(native, sp_flags, value):
  safety, _ = native
  safety.set_current_safety_param_sp(sp_flags)
  assert safety.set_safety_hooks(structs.CarParams.SafetyModel.hondaNidec, 0) == 0
  safety.init_tests()
  cruise(native, True)
  safety.set_controls_allowed(True)
  assert not safety.safety_tx_hook(button(native, value))


@pytest.mark.parametrize("bus", [0, 1, 3])
def test_no_extra_bus_permission(native, bus):
  safety, _ = native
  cruise(native, True)
  assert not safety.safety_tx_hook(button(native, CruiseButtons.RES_ACCEL, bus=bus))


@pytest.mark.parametrize("setting", [1, 2, 3])
def test_injected_settings_are_not_permitted(native, setting):
  safety, _ = native
  cruise(native, True)
  for value in (CruiseButtons.RES_ACCEL, CruiseButtons.DECEL_SET, CruiseButtons.CANCEL):
    assert not safety.safety_tx_hook(button(native, value, setting=setting))


def test_wrong_packet_length_rejected(native):
  safety, packer = native
  cruise(native, True)
  addr, data, bus = packer.make_can_msg("SCM_BUTTONS", 2, {"CRUISE_BUTTONS": CruiseButtons.RES_ACCEL})
  assert not safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, data + b"\x00" * 4))


@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize("aeb", [False, True])
def test_stock_forwarding_and_no_op_longitudinal_authority_are_unchanged(native, allowed, aeb):
  safety, packer = native
  cruise(native, True)
  safety.set_controls_allowed(allowed)
  assert safety.safety_rx_hook(packet(packer, "BRAKE_COMMAND", 2, {"COMPUTER_BRAKE": 40, "AEB_REQ_1": aeb}))
  # Injection must never take over/suppress the physical SCM stream.
  safety.safety_tx_hook(button(native, CruiseButtons.RES_ACCEL))
  for source, destination in ((0, 2), (2, 0)):
    for address in (0x296, 0x1FA, 0x30C):
      assert safety.safety_fwd_hook(source, address) == destination
  for command in (packet(packer, "ACC_HUD", 0, {"PCM_SPEED": 80, "PCM_GAS": 100}),
                  packet(packer, "BRAKE_COMMAND", 0, {"COMPUTER_BRAKE": 20})):
    assert not safety.safety_tx_hook(command)


def test_physical_cancel_wins(native):
  safety, _ = native
  cruise(native, True)
  assert safety.safety_rx_hook(button(native, CruiseButtons.CANCEL, bus=0))
  assert not safety.get_controls_allowed()
  assert not safety.safety_tx_hook(button(native, CruiseButtons.RES_ACCEL))
  assert safety.safety_fwd_hook(0, 0x296) == 2


def test_real_carcontroller_packet_passes_native_safety(native):
  from opendbc.car.honda.tests.test_stock_icbm import build, controls
  safety, _ = native
  cruise(native, True)
  ci = build()
  ci.CP_SP.pcmCruiseSpeed = False
  cc, sp = controls()
  ci.CC.frame = 6
  _, sends = ci.CC.update(cc.as_reader(), sp, ci.CS, 60_000_000)
  buttons = [msg for msg in sends if msg[0] == 0x296]
  assert len(buttons) == 1
  addr, data, bus = buttons[0]
  assert safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, data))
