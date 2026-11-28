"""Command timing for the Civic's non-interceptor Nidec PCM path."""
import math

from opendbc.car.honda.values import CAR, HondaFlags
from opendbc.car.honda.nidec_pcm_constraints import taper_positive_accel
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP


PCM_GAS_TICKS = 10  # ACC_HUD is transmitted at 10 Hz, from a 100 Hz controller.
REFERENCE_MAX_AGE_NS = 250_000_000
# Candidate preview for gentle slowdowns, not a learned inverse of engine drag.
# The positive-acceleration servo gain can be >10 s. Reusing it below zero
# turned a -0.1 m/s2 request into a multi-mph reduction of the PCM target.
SLOWDOWN_PREVIEW_SECONDS = 1.0


def coast_blend(requested: float, hill: float, speed: float, cruise: float, eligible: bool) -> tuple[float, float]:
  """Free-cruise road-load credit (m/s2) and positive speed-feedforward fraction.

  Candidate lower-envelope credit, not measured brake authority or a wind learner:
  Nov 8 Civic coast samples (no pedals/brake/reported gas, zero PCM target, >.5s
  settling) had -aEgo-g*sin(pitch) mostly above this 0.10--0.30 m/s2 curve.
  The recorded trajectory is NOT a closed-loop validation of this approximation.

  Credit fades out for overspeed or material slowdown. Leads, stops, invalid
  observations and overrides veto it at both producer and controller. No timer
  holds propulsion, no minimum speed target, no simultaneous gas/brake exception.
  The same credit reduces positive feedforward BEFORE friction is required, so
  gas cannot keep asking for full speed lead while the brake path enters coast.
  """
  if (not eligible or not all(math.isfinite(x) for x in (requested, hill, speed, cruise))
      or requested <= -.5 or not 5.0 < speed < 50.0 or not 5.0 < cruise <= 100.0):
    return 0.0, 1.0
  # Deliberately below the observed coast resistance; never apply the saved
  # Wind Brake factor (nearly 3 in the source drive, ~1.7 m/s2 at highway speed).
  points = ((5.0, .10), (13.0, .15), (22.0, .25), (35.0, .30))
  drag = points[-1][1]
  for (v0, d0), (v1, d1) in zip(points, points[1:], strict=False):
    if speed <= v1:
      drag = d0 + (d1 - d0) * (speed - v0) / (v1 - v0)
      break
  rolling = min((speed - 5.0) / 4.0, 1.0)
  # Full credit only through +0.25 m/s overspeed, none by +0.75 (~1.7 mph).
  speed_weight = min(max((.75 - (speed - cruise)) / .5, 0.0), 1.0)
  # A >=0.5 m/s2 slowdown retains the original brake path in full.
  demand_weight = min(max((requested + .5) / .35, 0.0), 1.0)
  credit = drag * rolling * min(speed_weight, demand_weight)
  # Do not restore positive feedforward at the zero-credit overspeed endpoint.
  # For negative wheel demand its limit as credit -> 0 is zero, not one.
  fraction = (min(max((requested + hill + credit) / credit, 0.0), 1.0) if credit > 0.0
              else float(requested + hill >= 0.0))
  return credit, fraction


class WindCreditTrial:
  """Bench-only: return some discarded brake demand when coasting isn't delivering it.

  Output is an additional wire count, not a pressure estimate. The limits are
  experiment bounds, not a measured Civic brake calibration.
  """
  def __init__(self):
    self.last_ns = 0
    self.seconds = 0.
    self.extra = 0.

  def update(self, now_ns, eligible, demand, measured, pre_wind_counts, base_counts):
    dt = (now_ns - self.last_ns) * 1e-9
    self.last_ns = now_ns
    if (not eligible or not 0. < dt <= .05 or not all(math.isfinite(x) for x in
        (demand, measured, pre_wind_counts, base_counts)) or demand >= 0.):
      self.seconds = self.extra = 0.
      return 0
    # Act only on a sustained tracking deficit, not merely a negative plan.
    deficit = measured - demand
    self.seconds = min(1., self.seconds + dt) if deficit > .20 else 0.
    headroom = max(0., pre_wind_counts - base_counts)
    target = min(12., 12. * max(0., deficit - .20), headroom) if self.seconds >= .5 else 0.
    self.extra = min(headroom, max(self.extra - 8. * dt, min(self.extra + 4. * dt, target)))
    return int(self.extra)


def reference_speed(CC_SP, now_nanos: int, requested_accel: float) -> float | None:
  """Extrapolate a fresh planner speed reference; never reuse stale targets."""
  age = now_nanos - CC_SP.civicPcmReferenceMonoTime
  speed = CC_SP.civicPcmReferenceSpeed
  if (CC_SP.civicPcmReferenceMonoTime <= 0 or not 0 <= age <= REFERENCE_MAX_AGE_NS or
      not math.isfinite(speed) or not 0.0 <= speed <= 100.0 or not math.isfinite(requested_accel)):
    return None
  return min(max(speed + requested_accel * (age / 1e9), 0.0), 100.0)


def speed_target(reference: float, requested_accel: float, acceleration_gain: float) -> float:
  """Keep acceleration authority without extrapolating its gain into coasting.

  Positive requests retain the previous feedforward exactly. A negative request
  previews at most one second of speed loss instead of using the learned
  propulsion gain as an inverse model of the PCM's very different torque-off
  response. This is a static, continuous map: no held target, release timer,
  current-speed floor, or delay of a falling planner reference. Gas/brake and
  freshness interlocks remain outside this helper and take precedence.
  """
  gain = acceleration_gain if requested_accel >= 0.0 else min(acceleration_gain, SLOWDOWN_PREVIEW_SECONDS)
  return min(max(reference + gain * requested_accel, 0.0), 100.0)


def speed_deadzone_target(target: float, speed: float, maximum: float) -> float:
  """Continuous target-speed deadband: zero width at 0 m/s, full width at 9 m/s.

  Inside the band request current speed, never a zero PCM target. This is a
  setpoint transform for the factory speed servo, not a replacement PID.
  """
  if (type(maximum) not in (int, float) or not math.isfinite(maximum) or not 0.0 <= maximum <= 0.2
      or not math.isfinite(target) or not math.isfinite(speed) or speed < 0.0):
    return target
  if maximum == 0.0:
    return target
  width = maximum * min(speed / 9.0, 1.0)
  error = target - speed
  return speed + math.copysign(max(abs(error) - width, 0.0), error)


def gas_allowance(requested_accel: float, actual_accel: float, speed_lead: float, maximum: int) -> float:
  """Legacy-style PCM authority ceiling, not an inverse pedal-position target.

  Leave full authority available while below the speed target. Close to the
  target, taper only for measured acceleration overshoot; below current speed,
  taper for speed overshoot too. These are the old override law's 0.3/1.1 m/s2
  and 0.5/1.0 m/s breakpoints, without its reduced low-speed authority ceiling.
  Braking/disengagement still zero the command outside this helper.
  """
  if not all(math.isfinite(v) for v in (requested_accel, actual_accel, speed_lead)) or requested_accel <= -0.5:
    # A material slowdown must release propulsion even before brake hysteresis
    # has ramped far enough to produce its first nonzero brake packet.
    return 0.0
  accel_error = actual_accel - requested_accel
  accel_limiter = 1.0 - 0.9 * min(max((accel_error - 0.3) / 0.8, 0.0), 1.0)
  # Acceleration overshoot should not starve a car still well below its target.
  accel_limiter = max(accel_limiter, min(max(speed_lead, 0.0), 1.0))
  speed_limiter = 1.0 - 0.9 * min(max(-speed_lead / 0.5, 0.0), 1.0)
  return float(maximum * min(speed_limiter, accel_limiter))


def allowance_command(target: float, previous_command: int, recovery: bool, maximum: int) -> int:
  """Keep the existing rise envelope, without inverse-model undershoot on release."""
  rise = 7 if recovery and previous_command < target else 2
  return int(min(max(target, 0.0), min(previous_command + rise, maximum)))


def uses_civic_pcm(CP, CP_SP) -> bool:
  return bool(CP.carFingerprint == CAR.HONDA_CIVIC and CP.openpilotLongitudinalControl and
              not (CP.flags & HondaFlags.BOSCH) and not CP_SP.enableGasInterceptor and
              not (CP_SP.flags & HondaFlagsSP.STOCK_LONGITUDINAL))


def speed_learning_observable(reference: float | None, speed: float, shaped_accel: float,
                              requested_accel: float, gas_wire: float, maximum: float) -> bool:
  """MVL's gain fit is valid only when the speed lead is effectively sf * accel.

  The planner-reference offset and comfort ramp can otherwise look like ECU lag.
  A full gas allowance leaves authority to the speed servo; a reduced allowance
  cannot distinguish servo response from the gas ceiling. Launch learning has
  its own measured-response gates and is independent of this steady-state fit.
  """
  return bool(reference is not None and all(math.isfinite(v) for v in (reference, speed, shaped_accel, requested_accel, gas_wire, maximum))
              and abs(reference - speed) <= 0.1 and shaped_accel > 0.05
              and abs(shaped_accel - taper_positive_accel(requested_accel, speed)) <= 0.005 and maximum > 0 and gas_wire >= maximum)


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


# Pedal-like transition (2026-11-28). Unvalidated on the road: the propulsion
# numbers come from emulator probes of the 37805-5AJ firmware, not from the car.
#
# The interceptor path is smooth because the brake waits for road load, the pedal
# reaches zero before any brake, and its controller closes the loop. On route
# 00000031--0fa3f0ec26 the speed-target law below sent the brake (and with it a
# zero PCM target) 5.1 times a minute, at 26 of 58 onsets against a planner
# request that was zero or positive. See nrdr/docs/civic_pcm_pedal_law_2026-11-28.md.

# Brake-side road load: the interceptor path's wind-brake curve at factor 1, as
# a brake fraction. The learned wind factor is not used (it has saved near 3).
ROAD_LOAD_BP = (0.0, 2.3, 35.0)  # m/s
ROAD_LOAD_FRACTION = (0.001, 0.002, 0.15)
# Acceleration the PCM reaches at PCM_GAS 200 in its acceleration mode (emulator,
# 60-122 km/h; the 30 km/h point is the calibration map's upper end).
ACCEL_CEILING_BP = tuple(kph / 3.6 for kph in (30.0, 60.0, 80.0, 91.0, 97.0, 110.0, 122.0))
ACCEL_CEILING = (0.76, 0.707, 0.630, 0.598, 0.565, 0.511, 0.489)  # m/s2
# The emulator probes cover highway speeds only: the previous law is kept below
# the first speed and blended out linearly up to the second.
PEDAL_LAW_SPEED = (6.0, 10.0)  # m/s
ACCEL_MODE_LEAD = 10.0 / 3.6  # m/s above speed; holds the PCM in its acceleration mode
SHED_GAIN = 10.0 / 3.6  # m/s of speed lead per m/s2 of acceleration error
SHED_LEAD = (-6.0 / 3.6, 1.4 / 3.6)  # m/s; inside the PCM's hold band, never its release threshold
SHED_ENTER = -0.05  # m/s2 of acceleration error
INTEGRAL_GAIN = 0.25  # 1/s
INTEGRAL_LIMIT = 0.15  # m/s2
# Speed-error term, as in the interceptor's Personal Pedal controller and on the same
# reference (the planner's present speed, which itself relaxes toward measured speed).
# Its Econ highway gain, proportional only: the PCM's 1-2 s lag leaves no margin for
# that controller's integral, and a standing offset from the target is accepted.
SPEED_ERROR_GAIN = 0.25  # m/s2 per m/s
SPEED_ERROR_LIMIT = 0.3  # m/s2
ACCEL_FILTER_SECONDS = 0.3
HARD_SLOWDOWN = -0.5  # m/s2; at or below this the throttle is released at once, as before


def _interp(x: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
  if x <= xs[0]:
    return ys[0]
  for (x0, y0), (x1, y1) in zip(zip(xs, ys, strict=True), zip(xs[1:], ys[1:], strict=True), strict=False):
    if x <= x1:
      return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
  return ys[-1]


def road_load_fraction(speed: float) -> float:
  """Brake fraction that lifting off already delivers; subtracted from the brake command."""
  return _interp(speed, ROAD_LOAD_BP, ROAD_LOAD_FRACTION) if math.isfinite(speed) else 0.0


def accel_ceiling(speed: float) -> float:
  return _interp(speed, ACCEL_CEILING_BP, ACCEL_CEILING)


class PedalLaw:
  """Acceleration command for the PCM with a progressive lift-off.

  Driving: PCM_SPEED sits a fixed lead above speed, which keeps the PCM in its
  acceleration mode, and PCM_GAS carries the requested acceleration as a share
  of the PCM's ceiling. A small clamped integral covers grade and map error.
  Shedding (a negative request, or acceleration above the request): PCM_GAS
  falls with the request and a small speed lead proportional to the error
  lowers the throttle through the PCM's hold mode. The lead stays inside the
  hold band, so the throttle is not cut. A request at or below HARD_SLOWDOWN
  releases at once. Braking and every interlock stay in the controller.
  """
  def __init__(self):
    self.accel = 0.0
    self.integral = 0.0
    self.shedding = False
    self.tick: int | None = None

  def reset(self, accel: float = 0.0) -> None:
    self.accel = accel if math.isfinite(accel) else 0.0
    self.integral = 0.0
    self.shedding = False

  def update(self, request: float, accel: float, speed: float, active: bool, braking: bool = False,
             dt: float = 0.01, gas_max: int = 198, tick: int | None = None,
             reference: float | None = None) -> tuple[float, float, float, bool]:
    """Returns (speed lead in m/s, PCM_GAS before the rise limit, blend weight, release).

    `tick` is the caller's frame counter. A gap in it (launch, disengagement) means
    the filter and integral are stale, so they restart from the present state.
    `reference` is the planner's present speed; the bounded speed-error term shapes
    propulsion only. The brake and the material-slowdown release use the raw request.
    """
    if tick is not None:
      stale = self.tick is not None and tick - self.tick != 1
      self.tick = tick
      if stale:
        self.reset(accel)
    if not active or not all(math.isfinite(x) for x in (request, accel, speed)):
      self.reset(accel)
      return 0.0, 0.0, 0.0, False
    self.accel += (accel - self.accel) * dt / (ACCEL_FILTER_SECONDS + dt)
    weight = min(max((speed - PEDAL_LAW_SPEED[0]) / (PEDAL_LAW_SPEED[1] - PEDAL_LAW_SPEED[0]), 0.0), 1.0)
    ceiling = accel_ceiling(speed)
    speed_term = 0.0
    if reference is not None and math.isfinite(reference):
      speed_term = min(max(SPEED_ERROR_GAIN * (reference - speed), -SPEED_ERROR_LIMIT), SPEED_ERROR_LIMIT)
    # Nothing beyond the PCM's reach is commanded or integrated.
    command = min(request + speed_term, ceiling)
    error = command - self.accel
    if command < 0.0 or error < SHED_ENTER:
      self.shedding = True
    elif error >= 0.0:
      self.shedding = False
    saturated = command + self.integral >= ceiling and error > 0.0
    if weight > 0.0 and not (braking or self.shedding or saturated):
      self.integral = min(max(self.integral + INTEGRAL_GAIN * error * dt, -INTEGRAL_LIMIT), INTEGRAL_LIMIT)
    gas = 0.0 if command < 0.0 else gas_max * min(max((command + self.integral) / ceiling, 0.0), 1.0)
    lead = min(max(SHED_GAIN * error, SHED_LEAD[0]), SHED_LEAD[1]) if self.shedding else ACCEL_MODE_LEAD
    return lead, gas, weight, request <= HARD_SLOWDOWN
