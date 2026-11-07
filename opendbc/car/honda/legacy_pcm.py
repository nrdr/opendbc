"""Pure Civic PCM command law, based on Kegman 05475df8 (0.7.8).

No Params, learners, wind/grade compensation, CAN, or host imports. The caller
owns eligibility, message scheduling and stock AEB arbitration. Inputs are SI
except brake (normalized travel); outputs retain existing Honda wire caps.
"""
from dataclasses import dataclass
import math

import numpy as np


KP_BP = (0.0, 5.0, 35.0)  # m/s
KP_V = (3.6, 2.4, 1.5)  # 1/s, speed error -> acceleration
KI_BP = (0.0, 35.0)  # m/s
KI_V = (0.54, 0.36)  # 1/s^2
DT = 0.01


def compute_gb(accel: float, speed: float) -> float:
  creep = max(0.0, (2.3 - speed) / 2.3) * 0.15
  return accel / 4.8 - creep


def brake_limit(speed: float) -> float:
  return float(np.interp(speed, (5.0, 20.0), (1.0, 0.8)))


def calc_accel_override(a_ego: float, a_target: float, v_ego: float, v_target: float) -> float:
  """Historical normalized PCM authority, NOT a pedal-position command.

  1.6 is the donor's normalization constant, not a cap on modern a_target.
  Keep the overshoot limiters before the final wire clamp, as in the donor.
  """
  e_v, e_a = v_ego - v_target, a_ego - a_target
  maximum = float(np.interp(v_ego, (10.0, 20.0), (0.714, 1.0)))
  speed_limiter = float(np.interp(e_v, (0.0, 0.5), (1.0, 0.1)))
  accel_limiter = max(float(np.interp(e_a, (0.3, 1.1), (1.0, 0.1))),
                      float(np.interp(e_v, (-1.0, 0.0), (1.0, 0.0))))
  return max(maximum, a_target / 1.6) * min(speed_limiter, accel_limiter)


def brake_hysteresis(brake: float, braking: bool, steady: float) -> tuple[float, bool, float]:
  if (brake < 0.02 and not braking) or brake < 0.005:
    brake = 0.0
  braking = brake > 0.0
  if brake == 0.0:
    steady = 0.0
  elif brake > steady + 0.01:
    steady = brake - 0.01
  elif brake < steady - 0.01:
    steady = brake + 0.01
  return steady, braking, steady


def brake_pump_hysteresis(brake: int, previous: int, pump_on: bool) -> bool:
  return True if brake > previous else False if brake < previous else pump_on


@dataclass(frozen=True)
class LegacyPCMCommand:
  pcm_speed: float = 0.0  # m/s, caller performs CAN unit conversion
  pcm_accel: int = 0  # PCM_GAS, 0..198
  apply_brake: int = 0  # COMPUTER_BRAKE, 0..255
  pump_on: bool = False
  brake_fraction: float = 0.0


class LegacyPCMController:
  """Call at 100 Hz; set send_brake on the actual 50 Hz brake-send ticks.

  Retains the host's 32-count/send brake-release bound and gas/brake exclusion.
  These are deliberate modern envelope guards, not donor PI retuning.
  """
  def __init__(self):
    self.reset()

  def reset(self) -> None:
    self.braking = False
    self.brake_steady = 0.0
    self.brake_last = 0.0
    self.apply_brake_last = 0
    self.pump_on = False

  def update(self, *, active: bool, v_ego: float, a_ego: float, v_target: float, a_target: float,
             brake: float, speed_offset: float = 0.0, send_brake: bool = True, cancel: bool = False,
             gas_pressed: bool = False, brake_pressed: bool = False, stock_aeb: bool = False,
             gas_max: int = 198, brake_max: int = 256) -> LegacyPCMCommand:
    finite = all(math.isfinite(x) for x in (v_ego, a_ego, v_target, a_target, brake, speed_offset))
    if not active or cancel or gas_pressed or brake_pressed or not finite:
      self.reset()
      return LegacyPCMCommand()

    brake = float(np.clip(brake, 0.0, brake_limit(v_ego)))
    demand, self.braking, self.brake_steady = brake_hysteresis(brake, self.braking, self.brake_steady)
    self.brake_last = float(np.clip(demand, self.brake_last - 2.0, self.brake_last + DT))
    scale = max(0, min(int(brake_max), 256))
    apply_brake = int(np.clip(self.brake_last * scale, 0, max(0, scale - 1)))
    if send_brake:
      apply_brake = min(max(0, scale - 1), max(self.apply_brake_last - 32, apply_brake))
      self.pump_on = brake_pump_hysteresis(apply_brake, self.apply_brake_last, self.pump_on)
      self.apply_brake_last = apply_brake

    discount = 1.0 - float(np.clip(brake * 3.0, 0.0, 1.0))
    pcm_speed = float(np.clip((v_target + speed_offset) * discount, 0.0, 100.0))
    pcm_accel = int(np.clip(calc_accel_override(a_ego, a_target, v_ego, v_target), 0.0, 1.0) * min(198, max(0, gas_max)))
    # Account for the most recently transmitted brake even on an odd tick.
    if apply_brake > 0 or self.apply_brake_last > 0 or stock_aeb:
      pcm_speed, pcm_accel = 0.0, 0
    return LegacyPCMCommand(pcm_speed, pcm_accel, apply_brake, self.pump_on, self.brake_last)
