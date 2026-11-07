"""Startup-scoped Civic PCM experiments; never a Panda safety override."""
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.honda.values import CAR
from opendbc.car.honda.nidec_pcm_constraints import uses_nidec_pcm

FIXED_CRUISE_MPS = 90.0 * CV.MPH_TO_MS


def qualified(CP, CP_SP, flag) -> bool:
  return bool(CP.carFingerprint == CAR.HONDA_CIVIC and uses_nidec_pcm(CP, CP_SP) and
              not (CP.passive or CP.dashcamOnly) and CP.flags & flag)


def governor_speed(fixed_cruise: bool, reported_speed: float) -> float:
  # The actual car-side ceiling, not the independent lower planner target.
  return FIXED_CRUISE_MPS if fixed_cruise else reported_speed
