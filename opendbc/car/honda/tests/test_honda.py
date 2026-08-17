import unittest

from opendbc.car import DT_CTRL, gen_empty_fingerprint, rate_limit, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, HondaFlags, HondaSafetyFlags


class TestHondaFingerprint(unittest.TestCase):
  def test_tja_bosch_only(self):
    for car_model in CAR:
      if car_model.config.flags & HondaFlags.BOSCH_TJA_CONTROL:
        assert car_model.config.flags & HondaFlags.BOSCH, "Nidec car found with TJA control"

  def test_clarity_params(self):
    def params(version=b""):
      fw = [structs.CarParams.CarFw(ecu="eps", fwVersion=version)] if version else []
      return CarInterface.get_params(CAR.HONDA_CLARITY, gen_empty_fingerprint(), fw, False, False, False)

    for clarity in (params(), params(b"39990-TRW,A020\x00\x00"), params(b"39990,TRW,A020\x00\x00")):
      self.assertEqual(clarity.flags & HondaFlags.HYBRID, HondaFlags.HYBRID)
      self.assertEqual(clarity.safetyConfigs[-1].safetyParam, HondaSafetyFlags.NIDEC_HYBRID)
      self.assertFalse(clarity.dashcamOnly)

  def test_linear_torque_params(self):
    cars = {
      CAR.HONDA_ACCORD: 4096,
      CAR.HONDA_CIVIC: 3840,
      CAR.HONDA_CIVIC_BOSCH: 4096,
      CAR.HONDA_CIVIC_BOSCH_DIESEL: 4096,
      CAR.HONDA_CLARITY: 3840,
      CAR.HONDA_CRV_5G: 4096,
      CAR.HONDA_INSIGHT: 4096,
    }
    bp = [0., 25. * CV.MPH_TO_MS - 1e-3, 25. * CV.MPH_TO_MS, 50. * CV.MPH_TO_MS]
    fw = [structs.CarParams.CarFw(ecu="eps", fwVersion=b"39990-TRW,A020\x00\x00")]
    for car, steer_max in cars.items():
      with self.subTest(car=car):
        cp = CarInterface.get_params(car, gen_empty_fingerprint(), fw, False, False, False)
        self.assertEqual((list(cp.lateralParams.torqueBP), list(cp.lateralParams.torqueV)), ([0, steer_max], [0, steer_max]))
        self.assertEqual([round(v, 3) for v in cp.lateralTuning.pid.kpBP], [round(v, 3) for v in bp])
        self.assertEqual([round(v, 3) for v in cp.lateralTuning.pid.kpV], [.018, .024, .048, .060])
        self.assertEqual([round(v, 3) for v in cp.lateralTuning.pid.kiBP], [round(v, 3) for v in bp])
        self.assertEqual([round(v, 3) for v in cp.lateralTuning.pid.kiV], [.006, .008, .016, .020])
        self.assertFalse(cp.dashcamOnly)
        self.assertEqual(cp.steerAtStandstill, car != CAR.HONDA_ACCORD)
        if car != CAR.HONDA_ACCORD:
          self.assertEqual(cp.minSteerSpeed, -1.)
    self.assertTrue(CarInterface.get_params(CAR.HONDA_CIVIC_2022, gen_empty_fingerprint(), fw, False, False, False).dashcamOnly)

  def test_linear_torque_filter(self):
    def controller(car):
      cp = CarInterface.get_non_essential_params(car)
      interface = CarInterface(cp)
      interface.CS.lkas_hud = False
      interface.CS.out.cruiseState.available = True
      return interface

    def step(interface, speed, torque=1., lat_active=True):
      interface.CS.out.vEgo = speed
      control = structs.CarControl()
      control.latActive = lat_active
      control.actuators.torque = torque
      return interface.apply(control.as_reader(), 0)[0].torque

    low_speed = controller(CAR.HONDA_CIVIC_BOSCH)
    self.assertAlmostEqual(step(low_speed, 49. * CV.MPH_TO_MS), 1. / 11.)
    self.assertEqual(step(low_speed, 49. * CV.MPH_TO_MS, lat_active=False), 0.)
    self.assertAlmostEqual(step(low_speed, 49. * CV.MPH_TO_MS), 1. / 11.)

    high_speed = controller(CAR.HONDA_CIVIC_BOSCH)
    self.assertAlmostEqual(step(high_speed, 51. * CV.MPH_TO_MS), .5)

    stock = controller(CAR.HONDA_CRV_HYBRID)
    expected = 0.
    for torque in (1., 1., -1.):
      expected = rate_limit(torque, expected, -3. * DT_CTRL, 3. * DT_CTRL)
      self.assertAlmostEqual(step(stock, 0., torque), expected)
