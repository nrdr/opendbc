"""Hardware eligibility shared by Honda's interface and parked settings UI."""
from opendbc.car.honda.values import CAR, HondaFlags


def civic_icbm_available(CP, CP_SP) -> bool:
  # Independent of longitudinal selection: parked users may preconfigure ICBM.
  # Actual factory-ACC ownership and driver/state guards belong at transmission.
  return bool(CP.carFingerprint == CAR.HONDA_CIVIC and CP.flags & HondaFlags.NIDEC and
              not CP.flags & HondaFlags.BOSCH and not CP_SP.enableGasInterceptor)
