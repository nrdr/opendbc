"""Optional Civic PCM coast experiment. Never owns the friction brakes."""
import math


FADE_SECONDS = 3.0
CONTEXT_MAX_AGE_NS = 300_000_000


def fresh_context(message, now_nanos: int) -> bool:
  stamp = message.pcmEngineBrakeMonoTime
  return bool(message.pcmEngineBrakeAllowed and stamp > 0 and 0 <= now_nanos - stamp <= CONTEXT_MAX_AGE_NS)


class EngineBrakeFade:
  def __init__(self):
    self.weight = 0.0
    self.last_time = None

  def reset(self):
    self.weight = 0.0
    self.last_time = None

  def apply(self, speed: float, gas: int, *, now_nanos: int, permitted: bool, request: bool) -> tuple[float, int]:
    if not permitted or not math.isfinite(speed) or speed < 0 or gas < 0:
      self.reset()
      return speed, gas
    now = now_nanos * 1e-9
    # A restart or delayed control frame is not time spent fading a command.
    dt = 0.0 if self.last_time is None else max(0.0, min(now - self.last_time, 0.05))
    self.last_time = now
    self.weight = max(0.0, min(1.0, self.weight + (1 if request else -1) * dt / FADE_SECONDS))
    if self.weight == 0.0:
      return speed, gas
    # Reduce both PCM authority and its speed target. A recovery can only
    # restore the current controller request, never stored pre-coast throttle.
    output_speed = speed * (1.0 - self.weight)
    output_gas = int(gas * (1.0 - self.weight))
    if output_gas > 0:
      output_speed = max(1.0, output_speed)  # Honda's existing positive-gas guard.
    return output_speed, output_gas
