import unittest

from opendbc.car import gen_empty_fingerprint, structs
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
    bp = [0., 11.175, 11.176, 22.352]
    fw = [structs.CarParams.CarFw(ecu="eps", fwVersion=b"39990-TRW,A020")]
    for car, steer_max in cars.items():
      with self.subTest(car=car):
        cp = CarInterface.get_params(car, gen_empty_fingerprint(), fw, False, False, False)
        self.assertEqual((list(cp.lateralParams.torqueBP), list(cp.lateralParams.torqueV)), ([0, steer_max], [0, steer_max]))
        self.assertEqual([round(v, 3) for v in cp.lateralTuning.pid.kpBP], bp)
        self.assertEqual([round(v, 3) for v in cp.lateralTuning.pid.kiBP], bp)
        self.assertEqual([round(v, 3) for v in cp.lateralTuning.pid.kpV], [.018, .024, .048, .060])
        self.assertEqual([round(v, 3) for v in cp.lateralTuning.pid.kiV], [.006, .008, .016, .020])
        self.assertAlmostEqual(cp.lateralTuning.pid.kf, 0.00006)
        self.assertFalse(cp.dashcamOnly)

    self.assertTrue(CarInterface.get_params(CAR.HONDA_CIVIC_2022, gen_empty_fingerprint(), fw, False, False, False).dashcamOnly)
    nbox = CarInterface.get_params(CAR.HONDA_NBOX_2G, gen_empty_fingerprint(), [], False, False, False)
    self.assertEqual(([round(v, 3) for v in nbox.lateralTuning.pid.kpV], [round(v, 3) for v in nbox.lateralTuning.pid.kiV]),
                     ([.6], [.18]))
