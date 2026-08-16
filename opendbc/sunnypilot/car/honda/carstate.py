from openpilot.common.params import Params

from opendbc.car.honda.values import CAR, STEER_THRESHOLD


class HondaCarStateFeatures:
  def __init__(self, CP):
    self.CP = CP
    self.params = Params()

  def steering_pressed(self, steering_torque: float, steering_angle: float) -> bool:
    stock_threshold = STEER_THRESHOLD.get(self.CP.carFingerprint, 1200)
    threshold = self._scaled_threshold("NrdrDriverOverrideThreshold", stock_threshold)

    try:
      center_angle = float(self.params.get("HondaCenterBoostThreshold"))
    except (TypeError, ValueError):
      center_angle = 0.0
    if center_angle > 0.0 and abs(steering_angle) <= center_angle:
      threshold = self._scaled_threshold("NrdrOverrideThresholdCenterBoost", stock_threshold, threshold)

    sensitive_eps = self.CP.carFingerprint in (CAR.HONDA_CLARITY, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH)
    if sensitive_eps and self.params.get_bool("NrdrIncreaseOverrideTolerance"):
      threshold *= 2
    return abs(steering_torque) > threshold

  def _scaled_threshold(self, key: str, stock_threshold: float, default=None) -> float:
    default = stock_threshold if default is None else default
    try:
      value = int(self.params.get(key))
    except (TypeError, ValueError):
      return default
    if value <= 0:
      return default
    return value if stock_threshold == 1200 else stock_threshold * value / 1200.0
