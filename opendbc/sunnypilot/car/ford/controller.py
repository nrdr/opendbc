from collections import deque

import numpy as np
from openpilot.cereal import messaging
from openpilot.common.params import Params

from opendbc.car import DT_CTRL
from opendbc.car.common.pid import PIDController
from opendbc.car.ford.values import CarControllerParams, FordFlags
from opendbc.car.lateral import MAX_LATERAL_ACCEL, apply_std_steer_angle_limits


def _limit_curvature(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, steering_angle, lat_active, CP, max_lat_accel=None):
  if v_ego_raw > 9:
    apply_curvature = np.clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                              current_curvature + CarControllerParams.CURVATURE_ERROR)

  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, steering_angle, lat_active, CarControllerParams.ANGLE_LIMITS)

  if CP.flags & FordFlags.CANFD:
    lat_accel_ceiling = MAX_LATERAL_ACCEL if max_lat_accel is None else max_lat_accel
    curvature_accel_limit = lat_accel_ceiling / (max(v_ego_raw, 1) ** 2)
    apply_curvature = float(np.clip(apply_curvature, -curvature_accel_limit, curvature_accel_limit))

  return apply_curvature

FORD_T_IDXS = [(i / 32.0) ** 2 * 10.0 for i in range(33)]
FORD_CURVATURE_LOOKUP_TIME = 0.42
FORD_PC_BLEND_RATIO = 0.40
FORD_CURV_RATE_DELTA_T = 0.3
FORD_CURV_RATE_SPEED_BP = [0.0, 14.5, 15.5]
FORD_CURV_RATE_SPEED_V = [1.0, 1.0, 0.0]
FORD_CURV_RATE_PC_BP = [0.0, 0.008, 0.01]
FORD_CURV_RATE_PC_V = [0.0, 0.0, 1.0]
FORD_LARGE_CURVE_FACTOR_BP = [0.001, 0.02]
FORD_LARGE_CURVE_FACTOR_V = [1.0, 0.80]
FORD_PATH_OFFSET_LOOKUP_TIME = 0.2
FORD_MIN_LANELINE_CONF_BP = [0.6, 0.8]
FORD_LC_PID_SPEED_BP = [0.0, 9.0, 15.0]
FORD_LC_PID_SPEED_V = [0.0, 0.0, 1.0]
FORD_LC_PATH_ANGLE_ROC_BP = [5, 15, 25]
FORD_LC_PATH_ANGLE_ROC_V = [0.003, 0.0015, 0.002]
FORD_LC_PATH_ANGLE_RESET_FRAMES = 30
FORD_LANE_CHANGE_FACTOR_BP = [4.4, 40.23]
FORD_LANE_CHANGE_FACTOR_V = [0.95, 0.85]
FORD_HUMAN_TURN_ANGLE = 45.0
FORD_PATH_ANGLE_MAX = 0.5
FORD_CURVATURE_MAX = 0.02
FORD_CURVATURE_RATE_MAX = 0.001023
FORD_POST_LC_FRAMES = 160
FORD_MAX_PATH_ANGLE_CHANGE = 0.00125
FORD_MAX_PATH_OFFSET_CHANGE = 0.00125
FORD_MAX_CURVATURE_RATE_CHANGE = 0.0001
FORD_LAT_PARAM_STEP = 50


def _ford_get_bool(params, key, default=False):
  try:
    value = params.get(key)
    if value is None:
      return default
    return params.get_bool(key)
  except Exception:
    return default


def _ford_get_float(params, key, default, min_value=None, max_value=None, scale=1.0):
  try:
    value = params.get(key)
  except Exception:
    value = None
  if value is None:
    ret = default
  else:
    try:
      if isinstance(value, bytes):
        value = value.decode("utf-8")
      ret = float(value) / scale
    except (AttributeError, TypeError, ValueError):
      ret = default
  if min_value is not None:
    ret = max(min_value, ret)
  if max_value is not None:
    ret = min(max_value, ret)
  return ret


class FordOemLateral:
  def __init__(self, CP):
    self.CP = CP
    self.apply_curvature_last = 0.0
    self.param_reader = Params()
    self._ford_lat_param_frame = 0
    self.ford_oem_lateral = False
    self.ford_human_turn = True
    self.ford_lane_positioning = True
    self.ford_lane_pos_gain = 1.0
    self.ford_max_lat_accel = MAX_LATERAL_ACCEL
    self._read_ford_lat_params()

    try:
      self.lat_sm = messaging.SubMaster(['modelV2'])
    except Exception:
      self.lat_sm = None
    self.model = None

    self.human_turn = False
    self.reset_steering_last = False
    self.post_reset_ramp_active = False
    self.lane_change = False
    self.lane_change_last = False
    self.post_lane_change_active = False
    self.post_lane_change_timer = 0
    self.pre_lane_change_values = {'path_angle': 0.0, 'path_offset': 0.0, 'desired_curvature_rate': 0.0}
    self.path_angle_last = 0.0
    self.lc_path_angle_reset_counter = 0
    deque_len = max(2, int(round(FORD_CURV_RATE_DELTA_T / (DT_CTRL * CarControllerParams.STEER_STEP))))
    self.curvature_rate_deque = deque(maxlen=deque_len)
    self.lc_pid = PIDController(k_p=0.25, k_i=0.05, rate=20)

  def _read_ford_lat_params(self):
    p = self.param_reader
    self.ford_oem_lateral = _ford_get_bool(p, "NrdrFordOemLateral", False)
    self.ford_human_turn = _ford_get_bool(p, "NrdrFordHumanTurn", True)
    self.ford_lane_positioning = _ford_get_bool(p, "NrdrFordLanePositioning", True)
    self.ford_lane_pos_gain = _ford_get_float(p, "NrdrFordLanePosGain", 100.0, 0.0, 300.0) / 100.0
    self.ford_max_lat_accel = _ford_get_float(p, "NrdrFordMaxLatAccel", MAX_LATERAL_ACCEL, 2.0, 3.5)

  def _handle_post_lane_change_transition(self, path_angle, path_offset, desired_curvature_rate):
    if self.lane_change_last and not self.lane_change:
      self.post_lane_change_active = True
      self.post_lane_change_timer = 0
      self.pre_lane_change_values = {'path_angle': 0.0, 'path_offset': 0.0, 'desired_curvature_rate': 0.0}
    self.lane_change_last = self.lane_change
    if self.post_lane_change_active:
      self.post_lane_change_timer += 1
      pv = self.pre_lane_change_values
      new_path_angle = float(np.clip(path_angle, pv['path_angle'] - FORD_MAX_PATH_ANGLE_CHANGE, pv['path_angle'] + FORD_MAX_PATH_ANGLE_CHANGE))
      new_path_offset = float(np.clip(path_offset, pv['path_offset'] - FORD_MAX_PATH_OFFSET_CHANGE, pv['path_offset'] + FORD_MAX_PATH_OFFSET_CHANGE))
      new_curv_rate = float(np.clip(desired_curvature_rate,
                                    pv['desired_curvature_rate'] - FORD_MAX_CURVATURE_RATE_CHANGE,
                                    pv['desired_curvature_rate'] + FORD_MAX_CURVATURE_RATE_CHANGE))
      self.pre_lane_change_values = {'path_angle': new_path_angle, 'path_offset': new_path_offset, 'desired_curvature_rate': new_curv_rate}
      if self.post_lane_change_timer >= FORD_POST_LC_FRAMES:
        self.post_lane_change_active = False
      return new_path_angle, new_path_offset, new_curv_rate
    return path_angle, path_offset, desired_curvature_rate

  def _compute_ford_oem_lateral(self, CC, CS, actuators):
    if not CC.latActive:
      self.curvature_rate_deque.clear()
      self.lc_pid.reset()
      self.apply_curvature_last = 0.0
      self.path_angle_last = 0.0
      self.reset_steering_last = False
      self.post_reset_ramp_active = False
      return 0.0, 0.0, 0.0, 0.0, 0, 1

    desired_curvature = actuators.curvature
    current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
    steering_pressed = CS.out.steeringPressed
    steering_angle_deg = CS.out.steeringAngleDeg
    precision_type = 1

    if self.model is not None and len(self.model.orientation.x) >= 17:
      curvatures = np.array(self.model.orientationRate.z) / max(0.01, CS.out.vEgoRaw)
      predicted_curvature = float(np.interp(FORD_CURVATURE_LOOKUP_TIME, FORD_T_IDXS, curvatures))
    else:
      predicted_curvature = 0.0
    requested_curvature = (predicted_curvature * FORD_PC_BLEND_RATIO) + (desired_curvature * (1 - FORD_PC_BLEND_RATIO))

    lane_change_dir = 0
    if self.model is not None:
      self.lane_change = self.model.meta.laneChangeState in (1, 2, 3)
      lane_change_dir = self.model.meta.laneChangeDirection
    else:
      self.lane_change = False
    lane_change_factor = float(np.interp(CS.out.vEgoRaw, FORD_LANE_CHANGE_FACTOR_BP, FORD_LANE_CHANGE_FACTOR_V))
    if self.lane_change and lane_change_dir == 1 and requested_curvature < 0:
      requested_curvature *= lane_change_factor
      precision_type = 0
    elif self.lane_change and lane_change_dir == 2 and requested_curvature > 0:
      requested_curvature *= lane_change_factor
      precision_type = 0

    self.human_turn = steering_pressed and abs(steering_angle_deg) > FORD_HUMAN_TURN_ANGLE
    reset_steering = 1 if ((self.human_turn and self.ford_human_turn) or (CS.out.vEgoRaw < 0.1)) else 0
    if reset_steering == 1:
      requested_curvature = 0.0

    apply_curvature = _limit_curvature(requested_curvature, self.apply_curvature_last, current_curvature,
                                       CS.out.vEgoRaw, 0., CC.latActive, self.CP, max_lat_accel=self.ford_max_lat_accel)

    if reset_steering == 1:
      apply_curvature = 0.0
      self.post_reset_ramp_active = False
    elif self.reset_steering_last and not reset_steering:
      self.post_reset_ramp_active = True
      self.apply_curvature_last = 0.0
    if self.post_reset_ramp_active:
      apply_curvature = apply_std_steer_angle_limits(requested_curvature, self.apply_curvature_last, CS.out.vEgoRaw,
                                                     0, CC.latActive, CarControllerParams.ANGLE_LIMITS)
      if abs(requested_curvature - apply_curvature) < max(abs(requested_curvature) * 0.1, 0.001):
        self.post_reset_ramp_active = False
    self.reset_steering_last = (reset_steering == 1)

    self.curvature_rate_deque.append(predicted_curvature)
    desired_curvature_rate = 0.0
    if len(self.curvature_rate_deque) > 1:
      delta_t = (FORD_CURV_RATE_DELTA_T if len(self.curvature_rate_deque) == self.curvature_rate_deque.maxlen
                 else (len(self.curvature_rate_deque) - 1) * (DT_CTRL * CarControllerParams.STEER_STEP))
      desired_curvature_rate = (self.curvature_rate_deque[-1] - self.curvature_rate_deque[0]) / delta_t / max(0.01, CS.out.vEgoRaw)
    desired_curvature_rate *= float(np.interp(abs(predicted_curvature), FORD_CURV_RATE_PC_BP, FORD_CURV_RATE_PC_V))
    desired_curvature_rate *= float(np.interp(CS.out.vEgoRaw, FORD_CURV_RATE_SPEED_BP, FORD_CURV_RATE_SPEED_V))
    desired_curvature_rate *= float(np.interp(abs(predicted_curvature), FORD_LARGE_CURVE_FACTOR_BP, FORD_LARGE_CURVE_FACTOR_V))
    if self.lane_change:
      desired_curvature_rate = 0.0

    path_offset = 0.0
    if self.model is not None:
      path_offset_position = float(np.interp(FORD_PATH_OFFSET_LOOKUP_TIME, FORD_T_IDXS, self.model.position.y))
      path_offset_lanelines = (self.model.laneLines[1].y[0] + self.model.laneLines[2].y[0]) / 2
      laneline_width = self.model.laneLines[2].y[0] + (-self.model.laneLines[1].y[0])
      laneline_width_tol = float(np.interp(laneline_width, [3.75, 4.25], [0.81, 0.59]))
      laneline_conf = min(self.model.laneLineProbs[1], self.model.laneLineProbs[2], laneline_width_tol)
      laneline_scale = float(np.interp(laneline_conf, FORD_MIN_LANELINE_CONF_BP, [0.0, 1.0]))
      path_offset = path_offset_position * (1 - laneline_scale) + path_offset_lanelines * laneline_scale
    if self.lane_change:
      path_offset = 0.0

    path_offset_error = path_offset * self.ford_lane_pos_gain
    path_offset_error *= float(np.interp(CS.out.vEgoRaw, FORD_LC_PID_SPEED_BP, FORD_LC_PID_SPEED_V))
    if not self.ford_lane_positioning:
      path_offset_error = 0.0
    path_angle = self.lc_pid.update(path_offset_error)
    if not self.ford_lane_positioning or reset_steering == 1:
      path_angle = 0.0
    path_angle_roc = float(np.interp(abs(CS.out.vEgoRaw), FORD_LC_PATH_ANGLE_ROC_BP, FORD_LC_PATH_ANGLE_ROC_V))
    path_angle = float(np.clip(path_angle, self.path_angle_last - path_angle_roc, self.path_angle_last + path_angle_roc))
    if steering_pressed:
      self.lc_path_angle_reset_counter += 1
    else:
      self.lc_path_angle_reset_counter = 0
    if self.lc_path_angle_reset_counter > FORD_LC_PATH_ANGLE_RESET_FRAMES:
      self.lc_pid.reset()

    path_angle, path_offset, desired_curvature_rate = self._handle_post_lane_change_transition(path_angle, path_offset, desired_curvature_rate)
    if reset_steering == 1:
      path_angle = 0.0

    apply_curvature = float(np.clip(apply_curvature, -FORD_CURVATURE_MAX, FORD_CURVATURE_MAX))
    desired_curvature_rate = float(np.clip(desired_curvature_rate, -FORD_CURVATURE_RATE_MAX, FORD_CURVATURE_RATE_MAX))
    path_angle = float(np.clip(path_angle, -FORD_PATH_ANGLE_MAX, FORD_PATH_ANGLE_MAX))
    path_offset = 0.0

    if reset_steering == 1:
      ramp_type = 3
      self.curvature_rate_deque.clear()
      self.lc_pid.reset()
    else:
      ramp_type = 2

    self.apply_curvature_last = apply_curvature
    self.path_angle_last = path_angle
    return apply_curvature, path_angle, path_offset, desired_curvature_rate, ramp_type, precision_type

  @property
  def enabled(self):
    return self.ford_oem_lateral and self.lat_sm is not None

  def limit_stock_curvature(self, curvature: float, v_ego: float) -> float:
    if not self.CP.flags & FordFlags.CANFD:
      return curvature
    limit = MAX_LATERAL_ACCEL / max(v_ego, 1.0) ** 2
    return float(np.clip(curvature, -limit, limit))

  def refresh(self):
    self._ford_lat_param_frame += 1
    if self._ford_lat_param_frame % FORD_LAT_PARAM_STEP == 0:
      self._read_ford_lat_params()
    if not self.enabled:
      return
    try:
      self.lat_sm.update(0)
      if self.lat_sm.updated['modelV2']:
        self.model = self.lat_sm['modelV2']
    except Exception:
      self.model = None

  def compute(self, CC, CS, actuators, previous_curvature: float):
    self.apply_curvature_last = previous_curvature
    return self._compute_ford_oem_lateral(CC, CS, actuators)
