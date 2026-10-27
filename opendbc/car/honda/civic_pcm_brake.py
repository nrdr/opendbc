"""Brake-only deceleration feedback for the speed-led Civic PCM controller.

The legacy road-load subtraction can clip a legitimate brake request to zero.
This regulator must be able to recover from that state. It never alters the
PCM acceleration request, speed reference, or propulsion calibration.
"""
import math


class CivicPCMBrakeFeedback:
  # Normalized brake / (m/s^2), and normalized brake / (m/s^2*s).
  # Kept separate from the legacy multiplicative brake-gain learner.
  KP = 0.15
  KI = 0.20
  ERROR_DEADBAND = 0.03  # m/s^2
  RISE_RATE = 0.5  # normalized brake / second; base braking is never slew limited here

  def __init__(self):
    self.reset()

  def reset(self):
    self.integral = 0.0
    self.feedback = 0.0
    self.active = False

  def update(self, requested: float, measured: float, base: float, *, allowed: bool,
             saturated: bool, maximum: float, dt: float, stopping: bool = False,
             previous: float = 0.0) -> float:
    if not allowed or not all(math.isfinite(v) for v in (requested, measured, base, maximum, dt)) or dt <= 0.0:
      self.reset()
      return base
    if not self.active:
      self.feedback = min(max(previous, 0.0), maximum)
    self.active = True

    error = measured - requested
    error = math.copysign(max(abs(error) - self.ERROR_DEADBAND, 0.0), error)
    proportional = self.KP * error
    base = min(max(base, 0.0), maximum)
    # Trim may be negative when feedforward overestimates the braking needed.
    # Rebase its bounds every tick: a trim learned against a large base must
    # never block recovery after that base falls to zero (or vice versa).
    lower, upper = -base, maximum - base
    self.integral = min(max(self.integral, lower), upper)
    proposed = min(max(self.integral + self.KI * error * dt, lower), upper)
    # Saturation must not accumulate a hidden braking demand. Unwinding remains
    # allowed; zero brake is deliberately NOT an integration freeze condition.
    trial = base + proportional + proposed
    if error > 0.0 and (saturated or trial > maximum):
      # A negative trim must still unwind when a positive P term saturates.
      if self.integral < 0.0:
        self.integral = min(proposed, 0.0)
    elif error < 0.0 and trial < 0.0:
      # Likewise, do not retain positive demand behind a clipped-zero output.
      if self.integral > 0.0:
        self.integral = max(proposed, 0.0)
    elif not (stopping and error > 0.0):
      self.integral = proposed
    target = min(max(base + proportional + self.integral, 0.0), maximum)
    # Do not slew-limit the base's existing response to a strong request. Only
    # extra feedback braking has this rise envelope; release is not held here.
    ceiling = self.feedback + .021 * dt if stopping else max(base, self.feedback) + self.RISE_RATE * dt
    self.feedback = min(target, ceiling)
    return self.feedback
