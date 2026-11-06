"""Explicit rolling cruise coasting; never a final brake-packet veto."""
import math

from opendbc.car import structs
from opendbc.car.honda.civic_pcm import uses_civic_pcm


FADE_SECONDS = 3.0
MAX_AGE_NS = 300_000_000
RECOVERY_HYSTERESIS_MS = 0.1  # ~0.22 mph below set; never extends the upper allowance.


def propulsion_recovery_allowed(CC, CS) -> bool:
  """A reduction-only gas handoff may outlive coast/brake permission.

  This never authorizes brake suppression. It prevents a denied coast context
  (including upper-bound jitter) from restoring hidden positive gas in one tick.
  Pedals, faults, AEB and near-stop control still bypass the comfort transition.
  """
  cs = CS.out
  return bool(CC.enabled and CC.longActive and cs.canValid and not cs.accFaulted and
              math.isfinite(cs.vEgo) and cs.vEgo > 5 and not cs.standstill and
              CC.actuators.longControlState == structs.CarControl.Actuators.LongControlState.pid and
              not (CC.cruiseControl.cancel or cs.gasPressed or cs.brakePressed or cs.stockAeb or
                   CS.stock_brake.get('AEB_REQ_1', 0) or CS.stock_brake.get('AEB_REQ_2', 0)))


def coast_permission(CP, CP_SP, CC, CC_SP, CS, now_ns: int, launching: bool) -> bool:
  stamp = CC_SP.pcmCruiseCoastMonoTime
  set_speed, upper = CC_SP.pcmCruiseCoastSetSpeed, CC_SP.pcmCruiseCoastMaxSpeed
  cs = CS.out
  values = (set_speed, upper, cs.vEgo, cs.vCruise, CC.actuators.accel)
  return bool(
    uses_civic_pcm(CP, CP_SP) and CC_SP.pcmCruiseCoastVersion == 1 and CC_SP.pcmCruiseCoastAllowed and
    stamp > 0 and 0 <= now_ns - stamp <= MAX_AGE_NS and
    all(math.isfinite(v) for v in values) and 5 < set_speed < 70 and 0 < upper - set_speed <= 10 * .44704 + 1e-5 and
    abs(cs.vCruise / 3.6 - set_speed) <= .01 and 5 < cs.vEgo <= upper and not cs.standstill and
    CC.enabled and CC.longActive and cs.canValid and not cs.accFaulted and not launching and
    CC.actuators.longControlState == structs.CarControl.Actuators.LongControlState.pid and CC.actuators.accel >= 0 and
    not (CC.driverMonitoringEscalation or CC.cruiseControl.cancel or cs.gasPressed or cs.brakePressed or cs.stockAeb or
         CC_SP.leadOne.status or CC_SP.leadTwo.status or
         CS.stock_brake.get('AEB_REQ_1', 0) or CS.stock_brake.get('AEB_REQ_2', 0) or CS.stock_brake.get('FCW', 0)))


class CruiseCoast:
  """Cap current requests using a transition anchored to the last held wire.

  Entry cannot uncover gas hidden by a friction-brake command. Recovery blends
  toward *current* demand, never replays stored throttle. A priority exclusion
  restores ordinary braking immediately. A reduction-only propulsion recovery
  may continue, but never grants permission to suppress brake demand.
  """
  def __init__(self):
    self.reset()

  def reset(self):
    self.phase = None
    self.progress = 0.0
    self.last_ns = None
    self.entry_speed = 0.0
    self.entry_gas = 0
    self.gas_ceiling = 0
    self.speed_ceiling = 0.0
    self.engine = False

  @property
  def active(self):
    return self.phase is not None

  @property
  def coasting(self):
    return self.phase == 'coast'

  def update(self, *, now_ns: int, permitted: bool, request: bool, engine: bool,
             held_speed: float, held_gas: int, braking: bool, recovery_allowed: bool = False) -> None:
    if (not math.isfinite(held_speed) or held_speed < 0 or held_gas < 0 or
        (self.last_ns is not None and now_ns <= self.last_ns)):
      self.reset()
      return
    if not permitted:
      if not self.active or not recovery_allowed:
        self.reset()
        return
      request = False  # Restore only propulsion; ordinary braking already owns this tick.
    dt = 0.0 if self.last_ns is None else min((now_ns - self.last_ns) * 1e-9, .05)
    self.last_ns = now_ns
    next_phase = 'coast' if request else ('recover' if self.active else None)
    if next_phase is None:
      return
    changed = next_phase != self.phase or (next_phase == 'coast' and engine != self.engine)
    if changed:
      previous_coast = self.coasting
      self.phase, self.engine = next_phase, engine
      self.entry_speed, self.entry_gas = held_speed, held_gas
      self.gas_ceiling = min(self.gas_ceiling, held_gas) if previous_coast and next_phase == 'coast' else held_gas
      self.speed_ceiling = held_speed
      self.progress = 0.0
    elif not braking:
      self.progress = min(1.0, self.progress + dt / FADE_SECONDS)
    if self.coasting and braking:
      # The last brake frame may release in this very update, before apply().
      # An older 10-Hz HUD must not become the entry throttle in that case.
      self.entry_speed, self.entry_gas, self.progress = 0.0, 0, 0.0
      self.gas_ceiling, self.speed_ceiling = 0, 0.0
    if self.phase == 'recover' and self.progress >= 1.0:
      self.reset()

  def apply(self, speed: float, gas: int, v_ego: float, *, braking: bool) -> tuple[float, int]:
    if not self.active:
      return speed, gas
    if braking:
      # Preserve brake-release tails. They own the existing zero-gas guard;
      # their release must not resurrect an earlier positive entry command.
      if self.coasting:
        self.entry_speed, self.entry_gas, self.progress = 0.0, 0, 0.0
        self.gas_ceiling, self.speed_ceiling = 0, 0.0
      else:
        self.reset()
      return speed, gas
    if not all(math.isfinite(v) for v in (speed, gas, v_ego)) or min(speed, gas, v_ego) < 0:
      self.reset()
      return speed, gas
    fraction = self.progress
    if self.coasting:
      target_speed = 0.0 if self.engine else v_ego
      speed_limit = self.entry_speed + (target_speed - self.entry_speed) * fraction
      gas_limit = self.entry_gas * (1.0 - fraction)
    else:
      speed_limit = self.entry_speed + (speed - self.entry_speed) * fraction
      gas_limit = self.entry_gas + (gas - self.entry_gas) * fraction
    result_speed = min(speed, max(0.0, speed_limit))
    result_gas = min(gas, max(0, int(gas_limit)))
    if self.coasting:
      result_gas = self.gas_ceiling = min(self.gas_ceiling, result_gas)
      if self.engine:
        result_speed = self.speed_ceiling = min(self.speed_ceiling, result_speed)
    if result_gas > 0 and result_speed < 1.0:
      # Respect the existing Honda positive-gas speed floor without increasing
      # a speed request that the current controller or coast transition denied.
      result_gas = 0
    return result_speed, result_gas
