import unittest
from collections import defaultdict

from opendbc.car import DT_CTRL, gen_empty_fingerprint, rate_limit, structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, CarControllerParams, HondaFlags, HondaSafetyFlags


class TestHondaFingerprint(unittest.TestCase):
  def test_tja_bosch_only(self):
    for car_model in CAR:
      if car_model.config.flags & HondaFlags.BOSCH_TJA_CONTROL:
        assert car_model.config.flags & HondaFlags.BOSCH, "Nidec car found with TJA control"

  def test_clarity_params(self):
    for version in (b"", b"39990-TRW,A020\0\0", b"39990,TRW,A020\0\0"):
      fw = [structs.CarParams.CarFw(ecu="eps", fwVersion=version)] if version else []
      cp = CarInterface.get_params(CAR.HONDA_CLARITY, gen_empty_fingerprint(), fw, False, False, False)
      cp.carFw = fw
      self.assertTrue(cp.flags & HondaFlags.HYBRID)
      self.assertEqual(cp.safetyConfigs[-1].safetyParam, HondaSafetyFlags.NIDEC_HYBRID)
      self.assertFalse(cp.dashcamOnly)
      self.assertEqual(cp.lateralTuning.which(), "pid")
      self.assertEqual(CarControllerParams(cp).STEER_MAX, 3840)

  def test_modified_admission_is_exact(self):
    cases = [
      (CAR.HONDA_CIVIC, b"39990-TEG,A010", True, 3840),
      (CAR.HONDA_CIVIC, b"39990-TBA,A030", False, 2560),
      (CAR.HONDA_CIVIC_BOSCH, b"39990-TBA,C020", True, 4096),
      (CAR.HONDA_CIVIC_BOSCH, b"39990-TBA,C120", False, 4096),
      (CAR.HONDA_INSIGHT, b"39990-TXM,A040", True, 4096),
      (CAR.HONDA_CRV_5G, b"39990-TLA,A040", False, 3840),
      (CAR.HONDA_ACCORD, b"39990-TRW,A020", False, 4096),
    ]
    for model, version, admitted, transport in cases:
      with self.subTest(model=model, version=version):
        fw = [structs.CarParams.CarFw(ecu="eps", fwVersion=version + b"\0\0")]
        cp = CarInterface.get_params(model, gen_empty_fingerprint(), fw, False, False, False)
        cp.carFw = fw
        self.assertEqual(cp.dashcamOnly, not admitted)
        self.assertEqual(CarControllerParams(cp).STEER_MAX, transport)

  def test_stock_civic_keeps_stock_gain_and_transport(self):
    cp = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC)
    self.assertAlmostEqual(cp.lateralTuning.pid.kpV[0], 1.1, places=6)
    self.assertEqual(CarControllerParams(cp).STEER_MAX, 2560)
    self.assertEqual(cp.minSteerSpeed, 0.)

  def test_stock_rate_limit_is_retained_even_for_modified_civic(self):
    fw = [structs.CarParams.CarFw(ecu="eps", fwVersion=b"39990-TEG,A010\0\0")]
    cp = CarInterface.get_params(CAR.HONDA_CIVIC, gen_empty_fingerprint(), fw, False, False, False)
    cp.carFw = fw
    interface = CarInterface(cp)
    interface.CS.lkas_hud = False
    interface.CS.acc_hud = defaultdict(int)
    interface.CS.stock_brake = defaultdict(int)
    interface.CS.is_metric = False
    interface.CS.out.cruiseState.available = True
    control = structs.CarControl()
    control.latActive = True
    expected = 0.
    for torque in (1., 1., -1., 0.):
      control.actuators.torque = torque
      expected = rate_limit(torque, expected, -3. * DT_CTRL, 3. * DT_CTRL)
      output = interface.apply(control.as_reader(), 0)[0].torque
      self.assertAlmostEqual(output, expected, places=3)
