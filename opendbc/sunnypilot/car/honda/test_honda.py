"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import unittest

from opendbc.testing import parameterized

from opendbc.car import gen_empty_fingerprint
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.structs import CarParams
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR

CarFw = CarParams.CarFw


class TestHondaEpsMod(unittest.TestCase):

  @parameterized("car_name, fw", [(CAR.HONDA_CIVIC, b'39990-TBA,A030\x00\x00'), (CAR.HONDA_CIVIC, b'39990-TBA-A030\x00\x00'),
                                  (CAR.HONDA_CLARITY, b'39990-TRW-A020\x00\x00'), (CAR.HONDA_CLARITY, b'39990,TRW,A020\x00\x00')])
  def test_eps_mod_fingerprint(self, car_name, fw):
    fingerprint = gen_empty_fingerprint()
    car_fw = [CarFw(ecu="eps", fwVersion=fw)]

    CarInterface = interfaces[car_name]
    CP = CarInterface.get_params(car_name, fingerprint, car_fw, False, False, False)
    _ = CarInterface.get_params_sp(CP, car_name, fingerprint, car_fw, False, False, False)

    self.assertFalse(CP.dashcamOnly)


class TestHondaPidTune(unittest.TestCase):

  @parameterized("car_name", list(CAR))
  def test_universal_pid_gain_ramp(self, car_name):
    fingerprint = gen_empty_fingerprint()
    CarInterface = interfaces[car_name]
    CP = CarInterface.get_params(car_name, fingerprint, [], False, False, False)
    _ = CarInterface.get_params_sp(CP, car_name, fingerprint, [], False, False, False)

    expected = (
      (CP.lateralTuning.pid.kpBP, [0., 50. * CV.MPH_TO_MS]),
      (CP.lateralTuning.pid.kpV, [0.03, 0.06]),
      (CP.lateralTuning.pid.kiBP, [0., 50. * CV.MPH_TO_MS]),
      (CP.lateralTuning.pid.kiV, [0.01, 0.02]),
    )
    for actual, target in expected:
      self.assertEqual(len(actual), len(target))
      for actual_value, target_value in zip(actual, target, strict=True):
        self.assertAlmostEqual(actual_value, target_value, delta=1e-6)
