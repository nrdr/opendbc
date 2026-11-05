from types import ModuleType, SimpleNamespace
from dataclasses import asdict, replace
from pathlib import Path
import subprocess

import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.carcontroller import NIDEC_SERVO_LAG_FRAMES
from opendbc.car.honda.engine_braking import EngineBrakeFade, fresh_context
from opendbc.car.honda.tests.test_civic_pcm import control, interface
from opendbc.car.honda.tests.test_mvl_nrdr_integration import learned_state
from opendbc.car.honda.values import CAR, DBC


DT_NS = 10_000_000
HUD = 0x30C
BRAKE = 0x1FA

# The actual nested-repo HEAD when engine braking was introduced. Pin it so
# committing the feature cannot silently turn this into a same-code comparison.
PRE_ENGINE_BRAKE_REV = '2320f7e4ff87eb52ed27567d9ab8e87b2f58a939'


@pytest.fixture(scope='module')
def pre_engine_brake_controller():
  repo = Path(__file__).resolve().parents[4]
  source_path = 'opendbc/car/honda/carcontroller.py'
  source = subprocess.check_output(['git', '-C', str(repo), 'show', f'{PRE_ENGINE_BRAKE_REV}:{source_path}'])
  assert b'engine_brake' not in source
  # Compile the exact Git blob in an isolated namespace: no source rewriting,
  # monkeypatch of production imports, temporary checkout, or fake CAN packer.
  baseline = ModuleType('opendbc.car.honda._pre_engine_brake_controller')
  exec(compile(source, f'{PRE_ENGINE_BRAKE_REV}:{source_path}', 'exec'), baseline.__dict__)
  return baseline.CarController


BASELINE_PATHS = [
  pytest.param(CAR.HONDA_CIVIC, False, 0, False, id='civic-mvl'),
  pytest.param(CAR.HONDA_CIVIC, False, 1, False, id='civic-ecu-matched'),
  pytest.param(CAR.HONDA_CIVIC, False, 2, True, id='civic-experimental-reference'),
  pytest.param(CAR.HONDA_CIVIC, False, 2, False, id='civic-experimental-compatibility'),
  pytest.param(CAR.HONDA_CLARITY, False, 0, False, id='clarity-pcm'),
  pytest.param(CAR.HONDA_CIVIC, True, 0, False, id='civic-interceptor'),
  pytest.param(CAR.HONDA_CLARITY, True, 0, False, id='clarity-interceptor'),
]


@pytest.mark.parametrize('car,interceptor,mode,reference', BASELINE_PATHS)
@pytest.mark.parametrize('learning', [False, True])
@pytest.mark.parametrize('phase', range(10))
def test_disabled_matches_prepatch_head_can(pre_engine_brake_controller, car, interceptor, mode, reference,
                                           learning, phase, record_property):
  actual, config = interface(car, interceptor=interceptor, learning=learning, pcm_mode=mode)
  expected, base_config = interface(car, interceptor=interceptor, learning=learning, pcm_mode=mode)
  expected.CC = pre_engine_brake_controller({bus: parser.dbc_name for bus, parser in expected.can_parsers.items()},
                                           expected.CP, expected.CP_SP, base_config.honda)
  assert type(actual.CC) is not type(expected.CC)
  assert not hasattr(expected.CC, 'engine_brake_fade')
  assert actual.CP.to_dict() == expected.CP.to_dict()
  assert asdict(actual.CP_SP) == asdict(expected.CP_SP)
  for ci in (actual, expected):
    ci.CC.average_factor = .035
    ci.CS.car_gas_available = True
    ci.CS.engine_rpm = 1500.
  decoder = CANParser(DBC[car][Bus.pt], [('ACC_HUD', 10), ('BRAKE_COMMAND', 50)], 0)
  seen_hud, positive_gas, positive_brakes = set(), set(), set()
  packet_count = byte_count = frame_count = 0
  # Deterministic input traces, not a claim about the physical Honda plant.
  # Every transition is shifted through all ten 100-Hz/10-Hz send phases.
  stages = [
    ('warmup', 100 + phase, .4, 20., 20., 'pid'),
    ('acceleration', 300, .8, 20., 22., 'pid'),
    ('taper', 120, .05, 22., 22., 'pid'),
    ('braking', 180, -2., 22., 2., 'pid'),
    ('stopping', 120, -1., 2., 0., 'stopping'),
    ('standstill', 100, -1., 0., 0., 'stopping'),
    ('starting', 60, .8, 0., .5, 'starting'),
    ('relaunch', 180, .8, .5, 8., 'pid'),
    ('gas_pedal', 40, .5, 8., 8., 'pid'),
    ('brake_pedal', 40, -1., 8., 7., 'pid'),
    ('aeb', 40, -2., 7., 5., 'pid'),
    ('cancel', 40, .2, 5., 5., 'pid'),
    ('inactive', 40, 0., 5., 5., 'off'),
    ('reengage', 140, .6, 5., 6., 'pid'),
  ]
  for stage, count, accel, start_speed, end_speed, state in stages:
    for index in range(count):
      now = (frame_count + 100) * DT_NS
      speed = start_speed + (end_speed - start_speed) * index / max(1, count - 1)
      outputs = []
      for ci in (actual, expected):
        cs = ci.CS.out
        cs.vEgo, cs.aEgo = speed, accel * .5
        cs.standstill = speed == 0.
        cs.canValid, cs.accFaulted = True, False
        cs.gearShifter = structs.CarState.GearShifter.drive
        cs.gasPressed, cs.brakePressed, cs.stockAeb = stage == 'gas_pedal', stage == 'brake_pedal', stage == 'aeb'
        cs.cruiseState.enabled = stage not in ('cancel', 'inactive')
        ci.CS.car_gas = 30. + frame_count % 11 if accel > 0. else 0.
        cc = control(accel)
        cc.enabled = cc.longActive = stage not in ('cancel', 'inactive', 'gas_pedal', 'brake_pedal')
        cc.latActive = cc.enabled
        cc.actuators.torque = .1 if (frame_count // 50) % 2 else -.1
        cc.actuators.longControlState = getattr(structs.CarControl.Actuators.LongControlState, state)
        cc.cruiseControl.cancel = stage == 'cancel'
        cc.hudControl.setSpeed = 30.
        sp = structs.CarControlSP(civicPcmEnabled=reference, civicPcmReferenceSpeed=speed + .5,
                                 civicPcmReferenceMonoTime=now if reference else 0)
        assert not sp.pcmEngineBrakeAllowed and not sp.pcmEngineBrakeRequest and sp.pcmEngineBrakeMonoTime == 0
        outputs.append(ci.apply(cc.as_reader(), sp, now))
      (actual_out, actual_can), (base_out, base_can) = outputs
      where = f'{car} mode={mode} interceptor={interceptor} learning={learning} phase={phase} stage={stage} frame={frame_count}'
      # Equality includes every address, bus, payload byte, message order and
      # send cadence, not merely the decoded longitudinal signal values.
      assert actual_can == base_can, where
      assert actual_out.to_dict() == base_out.to_dict(), where
      assert learned_state(actual.CC) == learned_state(expected.CC), where
      assert actual.CC.prior_gas_average == expected.CC.prior_gas_average, where
      assert actual.CC.average_factor_sens == expected.CC.average_factor_sens, where
      assert list(actual.CC.speed_cmd_hist) == list(expected.CC.speed_cmd_hist), where
      assert config.honda.provider.persisted == base_config.honda.provider.persisted, where
      assert actual.CC.engine_brake_fade.weight == actual.CC.engine_brake_washout == 0, where
      assert not actual.CC.engine_brake_wire_shaped, where
      decoder.update([(now, actual_can)])
      if any(addr == HUD for addr, _, _ in actual_can):
        seen_hud.add(stage)
        if decoder.vl['ACC_HUD']['PCM_GAS'] > 0:
          positive_gas.add(stage)
      if any(addr == BRAKE for addr, _, _ in actual_can) and decoder.vl['BRAKE_COMMAND']['COMPUTER_BRAKE'] > 0:
        positive_brakes.add(stage)
      assert any(addr == 0x200 for addr, _, _ in actual_can) == (interceptor and frame_count % 2 == 0), where
      packet_count += len(actual_can)
      byte_count += sum(len(data) for _, data, _ in actual_can)
      frame_count += 1
  assert {'acceleration', 'braking', 'standstill', 'relaunch', 'inactive'} <= seen_hud
  assert {'braking', 'standstill'} <= positive_brakes
  if not interceptor:
    assert {'acceleration', 'relaunch'} <= positive_gas
  record_property('baseline_revision', PRE_ENGINE_BRAKE_REV)
  record_property('compared_frames', frame_count)
  record_property('compared_can_packets', packet_count)
  record_property('compared_payload_bytes', byte_count)


class ObservedFade(EngineBrakeFade):
  """Record the real final-stage inputs, without changing its state or output."""
  def apply(self, speed, gas, **kwargs):
    self.input = (speed, gas)
    self.permitted = kwargs['permitted']
    self.output = super().apply(speed, gas, **kwargs)
    return self.output


class ControllerRig:
  def __init__(self, mode, phase=0):
    self.ci, self.config = interface(pcm_mode=mode)
    self.mode = mode
    self.ci.CS.out.canValid = True
    self.ci.CS.out.aEgo = .8
    self.ci.CS.car_gas_available = True
    self.ci.CS.car_gas = 30.
    self.ci.CS.engine_rpm = 1500.
    self.ci.CC.average_factor = .035
    self.ci.CC.engine_brake_fade = ObservedFade()
    self.decoder = CANParser(DBC[CAR.HONDA_CIVIC][Bus.pt], [('ACC_HUD', 10), ('BRAKE_COMMAND', 50)], 0)
    self.wire_gas = 0
    self.wire_speed = 0.
    for _ in range(100 + phase):
      self.step(allowed=False)
    assert self.ci.CC.frame % 10 == phase

  def enable_learning(self):
    provider = self.config.honda.provider
    provider.live_tuning = replace(provider.live_tuning, live_learning_gas=True)

  def step(self, *, allowed=True, request=True, reason=None, accel=.8):
    controller, cs = self.ci.CC, self.ci.CS.out
    frame = controller.frame
    now = (100 + frame) * DT_NS
    if reason == 'friction':
      # Test an actual friction-brake demand, not the unrelated ECU-Matched
      # acceleration ramp/sign-change coast interval before brake demand.
      controller.ecu_constraints.accel = -2.
      controller.ecu_constraints.last_sign = -1
      controller.ecu_constraints.coast_frames = 0
    cc = control(-2. if reason == 'friction' else accel)
    sp = structs.CarControlSP(pcmEngineBrakeAllowed=allowed, pcmEngineBrakeRequest=request,
                             pcmEngineBrakeMonoTime=now if allowed else 0,
                             civicPcmEnabled=self.mode == 2, civicPcmReferenceSpeed=21., civicPcmReferenceMonoTime=now)
    cs.gasPressed = reason == 'gas'
    cs.brakePressed = reason == 'brake'
    cs.stockAeb = reason == 'aeb'
    cs.canValid = reason != 'invalid_can'
    cs.accFaulted = reason == 'acc_fault'
    cc.cruiseControl.cancel = reason == 'cancel'
    cc.longActive = reason != 'inactive'
    if reason in ('stopping', 'starting'):
      cc.actuators.longControlState = reason
    if reason in ('leadOne', 'leadTwo'):
      getattr(sp, reason).status = True
    if reason == 'stale':
      sp.pcmEngineBrakeMonoTime = now - 300_000_001
    elif reason == 'future':
      sp.pcmEngineBrakeMonoTime = now + 1
    elif reason == 'missing':
      sp.pcmEngineBrakeAllowed = sp.pcmEngineBrakeRequest = False
      sp.pcmEngineBrakeMonoTime = 0

    before = SimpleNamespace(average=controller.prior_gas_average, sensitivity=controller.average_factor_sens,
                             alpha=controller.average_factor, gas=self.wire_gas, speed=self.wire_speed,
                             shaped=controller.engine_brake_wire_shaped, weight=controller.engine_brake_fade.weight,
                             lag_eligible=controller.speed_cmd_hist[0][2])
    out, packets = self.ci.apply(cc.as_reader(), sp, now)
    self.decoder.update([(now, packets)])
    hud = dict(self.decoder.vl['ACC_HUD']) if any(addr == HUD for addr, _, _ in packets) else None
    brake = dict(self.decoder.vl['BRAKE_COMMAND']) if any(addr == BRAKE for addr, _, _ in packets) else None
    if hud is not None:
      self.wire_gas, self.wire_speed = hud['PCM_GAS'], hud['PCM_SPEED'] / 3.6
    return SimpleNamespace(frame=frame, before=before, hud=hud, brake=brake, packets=packets, out=out,
                           weight=controller.engine_brake_fade.weight, washout=controller.engine_brake_washout,
                           raw=controller.engine_brake_fade.input, shaped=controller.engine_brake_fade.output)

  def assert_held_wire_observer(self, tick):
    controller, before = self.ci.CC, tick.before
    if tick.washout > 0 or self.mode == 2:
      alpha = min(max(before.alpha, .02), 1.)
      # Independent recurrence, driven by the PREVIOUS decoded packet, not
      # the proposed output or a controller telemetry field.
      assert controller.prior_gas_average == pytest.approx(before.average * (1. - alpha) + before.gas * alpha)
      assert controller.average_factor_sens == pytest.approx((before.gas - before.average) + (1. - alpha) * before.sensitivity)
    assert controller.pcm_gas_sent == self.wire_gas
    assert controller.pcm_speed_sent == pytest.approx(self.wire_speed, abs=.0051 / 3.6)
    if tick.hud is not None:
      assert tick.hud['PCM_GAS'] == tick.shaped[1]
      assert tick.hud['PCM_SPEED'] == pytest.approx(tick.shaped[0] * 3.6, abs=.0051)
      if tick.washout > 0 or self.mode == 2:
        assert tick.out.gas == pytest.approx(self.wire_gas / controller.params.NIDEC_GAS_MAX)


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('phase', range(10))
def test_controller_fade_recovery_and_decoded_held_wire(mode, phase):
  rig = ControllerRig(mode, phase)
  rig.enable_learning()
  controller = rig.ci.CC
  frozen = learned_state(controller)
  controller.sat_excess_frames, controller.sat_deficit_frames = 120, 180
  for index in range(301):
    tick = rig.step()
    rig.assert_held_wire_observer(tick)
    assert tick.weight == pytest.approx(index / 300.)
    # Regression: never feed the faded result back into the next tick's
    # previous-command limiter/inverse mapping via self.new_accel.
    assert controller.new_accel == tick.raw[1]
    assert tick.shaped[1] == (int(tick.raw[1] * (1. - tick.weight)) if tick.weight > 0 else tick.raw[1])
    assert learned_state(controller) == frozen
    assert not controller.speed_cmd_hist[-1][2]
    assert controller.sat_excess_frames == controller.sat_deficit_frames == 0
  assert controller.new_accel > 0
  assert tick.shaped[1] == 0

  for index in range(301):
    tick = rig.step(request=False)
    rig.assert_held_wire_observer(tick)
    assert tick.weight == pytest.approx(max(0., 1. - (index + 1) / 300.), abs=1e-12)
    assert controller.new_accel == tick.raw[1]
    assert learned_state(controller) == frozen
  assert tick.weight == 0
  assert tick.shaped == tick.raw
  # Observe the actual first restored packet, regardless of start phase.
  while tick.hud is None:
    tick = rig.step(request=False)
    rig.assert_held_wire_observer(tick)
  assert tick.hud['PCM_GAS'] == controller.new_accel > 0
  assert not controller.engine_brake_wire_shaped


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('phase', range(10))
def test_washout_starts_after_last_shaped_packet_is_replaced(mode, phase):
  rig = ControllerRig(mode, phase)
  rig.enable_learning()
  controller = rig.ci.CC
  frozen = learned_state(controller)
  for _ in range(151):
    rig.assert_held_wire_observer(rig.step())
  assert controller.engine_brake_wire_shaped
  restored_frame = None
  for _ in range(NIDEC_SERVO_LAG_FRAMES + 12):
    tick = rig.step(allowed=False, request=False)
    rig.assert_held_wire_observer(tick)
    assert tick.weight == 0
    if restored_frame is None:
      assert tick.washout == NIDEC_SERVO_LAG_FRAMES
      if tick.hud is not None:
        restored_frame = tick.frame
        assert not controller.engine_brake_wire_shaped
    else:
      assert tick.washout == max(0, NIDEC_SERVO_LAG_FRAMES - (tick.frame - restored_frame))
    if tick.washout > 0:
      assert learned_state(controller) == frozen
      assert not controller.speed_cmd_hist[-1][2]
      assert controller.sat_excess_frames == controller.sat_deficit_frames == 0
    else:
      assert restored_frame is not None
      assert tick.frame - restored_frame == NIDEC_SERVO_LAG_FRAMES
      assert not tick.before.lag_eligible
      break
  else:
    pytest.fail('washout never completed after the restored HUD packet')


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('phase', range(10))
@pytest.mark.parametrize('reason', ['leadOne', 'leadTwo', 'gas', 'brake', 'aeb', 'cancel', 'friction',
                                   'stale', 'future', 'missing', 'inactive', 'stopping', 'starting', 'invalid_can', 'acc_fault'])
def test_controller_priority_exit_at_every_hud_phase(mode, phase, reason):
  rig = ControllerRig(mode, phase)
  for _ in range(150):
    rig.step()
  assert rig.ci.CC.frame % 10 == phase
  assert rig.ci.CC.engine_brake_fade.weight > .4
  saw_hud, saw_brake = False, False
  for _ in range(10):
    tick = rig.step(reason=reason)
    rig.assert_held_wire_observer(tick)
    assert tick.weight == 0
    assert not rig.ci.CC.engine_brake_fade.permitted
    assert tick.raw == tick.shaped
    if tick.hud is not None:
      saw_hud = True
      assert tick.hud['PCM_GAS'] == rig.ci.CC.new_accel
    if reason == 'friction' and tick.brake is not None:
      saw_brake |= tick.brake['COMPUTER_BRAKE'] > 0
  assert saw_hud
  if reason == 'friction':
    assert saw_brake


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('phase', range(10))
def test_default_off_matches_denied_request_through_real_controller(mode, phase):
  off, denied = ControllerRig(mode, phase), ControllerRig(mode, phase)
  for index in range(250):
    reason = ('gas' if index < 50 else 'friction' if index < 100 else 'leadOne' if index < 150 else None)
    a = off.step(allowed=False, request=False, reason=reason)
    b = denied.step(allowed=False, request=True, reason=reason)
    assert a.packets == b.packets
    assert a.out.to_dict() == b.out.to_dict()
    assert learned_state(off.ci.CC) == learned_state(denied.ci.CC)
    assert off.ci.CC.prior_gas_average == denied.ci.CC.prior_gas_average
    assert off.ci.CC.average_factor_sens == denied.ci.CC.average_factor_sens
    assert a.weight == b.weight == a.washout == b.washout == 0


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('phase', range(10))
def test_friction_brake_packets_are_not_modified_by_fade_exit(mode, phase):
  faded, ordinary = ControllerRig(mode, phase), ControllerRig(mode, phase)
  for _ in range(150):
    faded.step()
    ordinary.step(allowed=False, request=False)
  positive_brakes = 0
  for _ in range(50):
    a = faded.step(reason='friction')
    b = ordinary.step(allowed=False, request=False, reason='friction')
    assert [p for p in a.packets if p[0] == BRAKE] == [p for p in b.packets if p[0] == BRAKE]
    if a.brake is not None:
      positive_brakes += a.brake['COMPUTER_BRAKE'] > 0
    assert a.weight == 0
  assert positive_brakes > 0


def advance(fade, start, count, request=True, **kw):
  return [fade.apply(30.0, 150, now_nanos=(start + i) * 10_000_000, permitted=True, request=request, **kw)
          for i in range(count)]


def test_three_second_fade_and_recovery():
  fade = EngineBrakeFade()
  outputs = advance(fade, 100, 301)
  assert outputs[0] == (30.0, 150)
  assert outputs[150][0] == pytest.approx(15)
  assert outputs[-1][0] == pytest.approx(0, abs=1e-10)
  assert outputs[-1][1] == 0
  assert all(a[0] >= b[0] and a[1] >= b[1] for a, b in zip(outputs, outputs[1:], strict=False))
  recovered = advance(fade, 401, 300, request=False)
  assert recovered[-1] == pytest.approx((30, 150), abs=1)


def test_interrupted_fade_returns_current_not_old_target():
  fade = EngineBrakeFade()
  advance(fade, 100, 151)
  speed, gas = fade.apply(20, 40, now_nanos=2_510_000_000, permitted=True, request=False)
  assert 10 < speed < 11 and gas <= 21


@pytest.mark.parametrize('weight', [0, .1, .5, 1])
def test_priority_exit_never_waits_for_fade(weight):
  fade = EngineBrakeFade()
  fade.weight = weight
  assert fade.apply(0, 0, now_nanos=1, permitted=False, request=True) == (0, 0)
  assert fade.weight == 0
  assert fade.apply(25, 100, now_nanos=2, permitted=False, request=False) == (25, 100)


def test_delayed_frame_is_not_a_completed_fade():
  fade = EngineBrakeFade()
  advance(fade, 100, 1)
  speed, gas = fade.apply(30, 150, now_nanos=10_000_000_000, permitted=True, request=True)
  assert speed >= 29.5 and gas >= 147


@pytest.mark.parametrize('allowed,stamp,now,expected', [(True, 1, 1, True), (False, 1, 1, False),
  (True, 0, 1, False), (True, 2, 1, False), (True, 1, 300_000_001, True), (True, 1, 300_000_002, False)])
def test_context_freshness(allowed, stamp, now, expected):
  assert fresh_context(SimpleNamespace(pcmEngineBrakeAllowed=allowed, pcmEngineBrakeMonoTime=stamp), now) is expected


def test_positive_gas_always_has_positive_speed():
  fade = EngineBrakeFade()
  for frame in range(400):
    speed, gas = fade.apply(6, 198, now_nanos=frame * 10_000_000, permitted=True, request=True)
    assert 0 <= gas <= 198 and 0 <= speed <= 6
    assert gas == 0 or speed >= 1
