"""RDX rack admission and signed E4 transport; all CAN frames stay in memory."""
from dataclasses import replace

import pytest

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.fw_versions import match_fw_to_car_exact
from opendbc.car.honda.fingerprints import FW_VERSIONS
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from opendbc.sunnypilot.car.tests.runtime_config import make_test_car_config


PTM = b"39990-TJB,A220\x00\x00"
STOCK = b"39990-TJB-A220\x00\x00"
CHASSIS = [(CAR.ACURA_RDX_3G, 4095), (CAR.ACURA_RDX_3G_MMR, 4076)]


def reply(version=PTM, **fields):
  return structs.CarParams.CarFw(**({"ecu": "eps", "address": 0x18DA30F1, "fwVersion": version} | fields))


def params(car, replies, alpha_long=False):
  cfg = make_test_car_config()
  cfg.honda.provider.live_tuning = replace(cfg.honda.provider.live_tuning, override_fade_up_s=0.0)
  fp = gen_empty_fingerprint()
  cp = CarInterface.get_params(car, fp, replies, alpha_long=alpha_long, is_release=False, docs=False, interface_config=cfg)
  sp = CarInterface.get_params_sp(cp, car, fp, replies, alpha_long=alpha_long, is_release_sp=False, docs=False)
  return cp, sp, cfg


@pytest.mark.parametrize("car,stock_max", CHASSIS)
@pytest.mark.parametrize("replies", [
  pytest.param([], id="missing"),
  pytest.param([reply(STOCK)], id="stock-a220"),
  pytest.param([reply(b"39990-TJB,A130\x00\x00")], id="unknown-mod"),
  pytest.param([reply(PTM.rstrip(b"\x00"))], id="missing-padding"),
  pytest.param([reply(PTM + b"\x00")], id="extra-padding"),
  pytest.param([reply(ecu="fwdCamera")], id="wrong-ecu"),
  pytest.param([reply(address=0x18DAB5F1)], id="wrong-address"),
  pytest.param([reply(subAddress=1)], id="wrong-subaddress"),
  pytest.param([reply(logging=True)], id="logging-only"),
  pytest.param([reply(STOCK), reply(b"other,version")], id="split-evidence"),
  pytest.param([reply(), reply(STOCK)], id="conflicting-stock"),
  pytest.param([reply(b"unknown"), reply()], id="conflicting-unknown"),
])
def test_unverified_rack_keeps_stock_map(car, stock_max, replies):
  cp, _, _ = params(car, replies)
  assert list(cp.lateralParams.torqueBP) == list(cp.lateralParams.torqueV) == [0, stock_max]


def test_exact_reply_on_older_rdx_keeps_stock_map():
  cp, _, _ = params(CAR.ACURA_RDX, [reply()])
  assert list(cp.lateralParams.torqueBP) == list(cp.lateralParams.torqueV) == [0, 1000]


@pytest.mark.parametrize("car", [car for car, _ in CHASSIS])
def test_rack_requires_an_independent_exact_chassis_match(car):
  ep = (0x18DA30F1, None)
  live = {key[1:]: {versions[0]} for key, versions in FW_VERSIONS[car].items()}
  assert match_fw_to_car_exact(live, "honda") == {car}  # stock fingerprint
  live[ep] = {PTM}
  assert match_fw_to_car_exact(live, "honda") == {car}
  assert not match_fw_to_car_exact({ep: {PTM}}, "honda")
  if car == CAR.ACURA_RDX_3G:
    live[ep] = {b"39990-TJB,A130\x00\x00"}
    assert car not in match_fw_to_car_exact(live, "honda")
  else:
    # MMR identifies from its original camera/radar; adding a new required EPS record would break stock cars.
    assert (structs.CarParams.Ecu.eps, *ep) not in FW_VERSIONS[car]


@pytest.mark.parametrize("car,stock_max", CHASSIS)
@pytest.mark.parametrize("alpha_long", [False, True])
def test_exact_rack_changes_only_mapping_and_existing_modified_flag(car, stock_max, alpha_long):
  stock_cp, stock_sp, _ = params(car, [reply(STOCK)], alpha_long)
  cp, sp, _ = params(car, [reply(), reply(b"serial", logging=True), reply(), reply(STOCK, ecu="fwdCamera")], alpha_long)
  expected = stock_cp.to_dict()
  assert expected["lateralParams"] == {"torqueBP": [0, stock_max], "torqueV": [0, stock_max]}
  expected["lateralParams"] = {"torqueBP": [0, 4096], "torqueV": [0, 4096]}
  # Includes lateral tuning, safety configs, minSteerSpeed, longitudinal and standstill policy.
  assert cp.to_dict() == expected
  assert sp == replace(stock_sp, flags=stock_sp.flags | HondaFlagsSP.EPS_MODIFIED)


@pytest.mark.parametrize("car,stock_max", CHASSIS)
@pytest.mark.parametrize("alpha_long", [False, True])
def test_controller_packs_signed_e4_and_keeps_inactive_heartbeat(car, stock_max, alpha_long):
  def make_interface(version):
    ci = CarInterface(*params(car, [reply(version)], alpha_long))
    ci.update([])
    ci.CS.out.vEgo = 25.0
    ci.CS.out.standstill = False
    ci.CS.steer_control_active = True
    return ci

  def apply(controller, limit, tick, enabled, active, output):
    cc = structs.CarControl(enabled=enabled, latActive=active)
    cc.actuators.torque = output
    out, frames = controller.apply(cc.as_reader(), structs.CarControlSP(), tick * 10_000_000)
    (bus, data), = [(b, d) for a, d, b in frames if a == 0xE4]
    assert bus == int(controller.CP.openpilotLongitudinalControl)
    assert bool(data[2] & 0x80) == active  # DBC STEER_TORQUE_REQUEST
    assert int.from_bytes(data[:2], "big", signed=True) == out.torqueOutputCan == (int(-output * limit) if active else 0)
    return frames

  ci, stock = make_interface(PTM), make_interface(STOCK)
  cases = [(True, True, 1.0), (True, True, -1.0), (True, True, 0.5), (True, False, 1.0), (False, False, -1.0)]
  for tick, (enabled, active, output) in enumerate(cases):
    mod_frames = apply(ci, 4096, tick, enabled, active, output)
    stock_frames = apply(stock, stock_max, tick, enabled, active, output)
    assert [(a, b) for a, _, b in mod_frames] == [(a, b) for a, _, b in stock_frames]
    if not active:
      assert mod_frames == stock_frames
