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

    stock, modified_2x, modified_3x = params(), params(b"39990-TRW,A020\x00\x00"), params(b"39990,TRW,A020\x00\x00")
    self.assertEqual(stock.flags & HondaFlags.HYBRID, HondaFlags.HYBRID)
    self.assertEqual(stock.safetyConfigs[-1].safetyParam, HondaSafetyFlags.NIDEC_HYBRID)
    self.assertEqual((list(stock.lateralParams.torqueBP), list(stock.lateralParams.torqueV)), ([0, 2560], [0, 2560]))
    self.assertEqual((list(modified_2x.lateralParams.torqueBP), list(modified_2x.lateralParams.torqueV)),
                     ([0, 0xA00, 0x2800], [0, 2560, 3840]))
    self.assertEqual((list(modified_3x.lateralParams.torqueBP), list(modified_3x.lateralParams.torqueV)),
                     ([0, 0xA00, 0x3C00], [0, 2560, 3840]))
