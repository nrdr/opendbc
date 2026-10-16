"""Command timing for the Civic's non-interceptor Nidec PCM path."""
import math

from opendbc.car.honda.values import CAR, HondaFlags
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP


PCM_GAS_TICKS = 10  # ACC_HUD is transmitted at 10 Hz, from a 100 Hz controller.


def uses_civic_pcm(CP, CP_SP) -> bool:
  return bool(CP.carFingerprint == CAR.HONDA_CIVIC and CP.openpilotLongitudinalControl and
              not (CP.flags & HondaFlags.BOSCH) and not CP_SP.enableGasInterceptor and
              not (CP_SP.flags & HondaFlagsSP.STOCK_LONGITUDINAL))


def response_per_tick(alpha: float) -> float:
  # A persisted value must not produce a singular/unstable inverse. This is the
  # same learner range, not a reset of stored tuning or a lower authority limit.
  return min(max(alpha, 0.02), 1.0) if math.isfinite(alpha) else 0.5


def gas_command(target: float, average: float, previous_command: int, alpha: float,
                recovery: bool, maximum: int) -> int:
  """Invert one complete sent-command interval, preserving existing rise limits.

  Inverting a 10 ms response and then holding that result for 100 ms magnified
  tiny reductions into zero-gas pulses. At alpha=.02 a 100->98 target produced
  0 followed by a half-second rise-limited recovery. The equivalent 100 ms
  response is ~.183, not .02. Keep the fractional target until final packing.
  """
  alpha_send = 1.0 - (1.0 - response_per_tick(alpha)) ** PCM_GAS_TICKS
  command = (target - average * (1.0 - alpha_send)) / alpha_send
  # Prepare at 100 Hz with exactly the old per-tick rise envelope. Only the
  # inverse horizon is 100 ms; multiplying the rise limit here would make the
  # first engage/override-release packet more aggressive than before.
  rise = 7 if recovery and previous_command < target else 2
  return int(min(max(command, 0.0), min(previous_command + rise, maximum)))


def advance_model(average: float, sensitivity: float, held_wire: int, alpha: float) -> tuple[float, float]:
  """Advance only the command actually on the wire, including between sends."""
  alpha = response_per_tick(alpha)
  return (average * (1.0 - alpha) + held_wire * alpha,
          (held_wire - average) + (1.0 - alpha) * sensitivity)
