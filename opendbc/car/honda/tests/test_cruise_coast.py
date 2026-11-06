"""Coast-transition and packed-controller regressions; no vehicle simulation."""
import math
import copy

import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.cruise_coast import CruiseCoast
from opendbc.car.honda.tests.test_civic_pcm import interface, control
from opendbc.car.honda.tests.test_mvl_nrdr_integration import learned_state
from opendbc.car.honda.values import CAR, DBC


def step(state, tick, *, request=True, engine=True, permitted=True, speed=25., gas=100, held_speed=25., held_gas=100, braking=False):
  state.update(now_ns=1_000_000_000 + tick * 10_000_000, permitted=permitted, request=request, engine=engine,
               held_speed=held_speed, held_gas=held_gas, braking=braking)
  return state.apply(speed, gas, 20., braking=braking)


@pytest.mark.parametrize('engine', [False, True])
def test_three_second_entry_and_current_demand_recovery(engine):
  state = CruiseCoast()
  outputs = [step(state, t, engine=engine) for t in range(302)]
  assert outputs[0] == (25., 100)
  assert outputs[150] == pytest.approx((12.5 if engine else 22.5, 50), abs=1)
  assert outputs[-1] == pytest.approx((0. if engine else 20., 0), abs=1e-9)
  assert all(a[1] >= b[1] for a, b in zip(outputs, outputs[1:], strict=False))
  for t in range(302, 604):
    result = step(state, t, request=False, engine=engine, speed=24., gas=80, held_speed=outputs[-1][0], held_gas=0)
    assert result[0] <= 24 and result[1] <= 80
  assert result == (24., 80) and not state.active


def test_no_positive_pulse_after_a_zero_base_request_or_toggle_change():
  state = CruiseCoast()
  step(state, 0)
  assert step(state, 1, speed=0, gas=0) == (0., 0)
  assert step(state, 2) == (0., 0)
  # Even if the last CAN tick predates this zero request, a live mode change
  # cannot lift the already-consumed gas ceiling while still coasting.
  assert step(state, 3, engine=False)[1] == 0
  assert step(state, 4, engine=True)[1] == 0


def test_brake_tail_does_not_resurrect_masked_gas():
  state = CruiseCoast()
  step(state, 0)
  assert step(state, 1, braking=True, gas=0, speed=0) == (0, 0)
  for t in range(2, 310):
    assert step(state, t, held_speed=0, held_gas=0) == (0., 0)


def test_priority_exclusion_resets_without_waiting_for_comfort_fade():
  state = CruiseCoast()
  for t in range(100):
    step(state, t)
  assert state.active
  assert step(state, 100, permitted=False) == (25., 100)
  assert not state.active


class Rig:
  def __init__(self, mode=1, *, engine=True, car=CAR.HONDA_CIVIC, interceptor=False, learning=False):
    self.ci, self.config = interface(car, interceptor=interceptor, pcm_mode=mode, learning=learning)
    self.ci.CS.out.canValid = True
    self.ci.CS.out.vCruise = 72.
    self.ci.CS.out.vEgo = 21.
    self.engine = engine
    self.tick = 0
    self.decoder = CANParser(DBC[car][Bus.pt], [('ACC_HUD', 10), ('BRAKE_COMMAND', 50)], 0)

  def frame(self, *, allowed=True, accel=0., pitch=-.07, mutate=None, set_speed=20.):
    now = (100 + self.tick) * 10_000_000
    cc = control(accel)
    cc.orientationNED = [0., pitch, 0.]
    sp = structs.CarControlSP(pcmCruiseCoastVersion=1, pcmCruiseCoastAllowed=allowed,
                             pcmCruiseCoastSetSpeed=set_speed, pcmCruiseCoastMaxSpeed=set_speed + 2.2352,
                             pcmCruiseCoastMonoTime=now, pcmEngineBrakeEnabled=self.engine)
    if mutate:
      mutate(cc, sp, self.ci.CS, self.ci.CC)
    out, packets = self.ci.apply(cc.as_reader(), sp, now)
    self.decoder.update([(now, packets)])
    self.tick += 1
    hud = dict(self.decoder.vl['ACC_HUD']) if any(a == 0x30C for a, _, _ in packets) else None
    brake = dict(self.decoder.vl['BRAKE_COMMAND']) if any(a == 0x1FA for a, _, _ in packets) else None
    return out, packets, hud, brake


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('engine', [False, True])
def test_downhill_coasts_without_friction_or_positive_propulsion(mode, engine):
  rig = Rig(mode, engine=engine, learning=True)
  c = rig.ci.CC
  # A retained negative hold integral and a learned hydraulic multiplier must
  # not manufacture braking for a fresh zero-accel cruise coast request.
  c.nidec_pid.i = c.nidec_pid_factor = -.04727
  c.nidec_brake_pid.i = c.brake_pid_factor = 1.66233
  before = learned_state(c)
  seen = 0
  for _ in range(350):
    out, _, hud, brake = rig.frame()
    assert c.cruise_coast.coasting
    assert c.nidec_pid.i == c.nidec_pid_factor == 0
    assert c.nidec_brake_pid.i == c.brake_pid_factor == 1.66233
    assert out.brake == 0
    if brake:
      seen += 1
      assert brake['COMPUTER_BRAKE'] == brake['BRAKE_LIGHTS'] == brake['COMPUTER_BRAKE_REQUEST'] == 0
    if hud:
      assert hud['PCM_GAS'] == 0  # Last held gas was zero; never uncover road-load compensation.
  assert seen > 100
  assert learned_state(c) == before
  assert c.pcm_speed_sent == pytest.approx(0 if engine else min(c.pcm_speed_sent, 21.))


@pytest.mark.parametrize('mode', [0, 1, 2])
def test_existing_brake_tail_releases_normally_before_coast_propulsion(mode):
  rig = Rig(mode)
  c = rig.ci.CC
  c.brake_last, c.brake_steady, c.braking = .8, .8, True
  c.apply_brake_last = 180
  c.pcm_gas_sent = 0
  counts = []
  for _ in range(130):
    _, _, hud, brake = rig.frame()
    if brake:
      counts.append(brake['COMPUTER_BRAKE'])
    if hud:
      assert hud['PCM_GAS'] == 0
  assert counts[0] >= 148  # Existing 32-count release limiter was not bypassed.
  assert all(a >= b for a, b in zip(counts, counts[1:], strict=False))
  assert counts[-1] == 0


EXCLUSIONS = [
  lambda cc, sp, cs, c: setattr(cc.actuators, 'accel', -.001),
  lambda cc, sp, cs, c: setattr(cc, 'driverMonitoringEscalation', True),
  lambda cc, sp, cs, c: setattr(cc.cruiseControl, 'cancel', True),
  lambda cc, sp, cs, c: setattr(cc, 'longActive', False),
  lambda cc, sp, cs, c: setattr(cc.actuators, 'longControlState', 'stopping'),
  lambda cc, sp, cs, c: setattr(cc.actuators, 'longControlState', 'starting'),
  lambda cc, sp, cs, c: setattr(cs.out, 'gasPressed', True),
  lambda cc, sp, cs, c: setattr(cs.out, 'brakePressed', True),
  lambda cc, sp, cs, c: setattr(cs.out, 'stockAeb', True),
  lambda cc, sp, cs, c: cs.stock_brake.update(AEB_REQ_1=1),
  lambda cc, sp, cs, c: cs.stock_brake.update(AEB_REQ_2=1),
  lambda cc, sp, cs, c: cs.stock_brake.update(FCW=1),
  lambda cc, sp, cs, c: setattr(cs.out, 'canValid', False),
  lambda cc, sp, cs, c: setattr(cs.out, 'accFaulted', True),
  lambda cc, sp, cs, c: setattr(cs.out, 'standstill', True),
  lambda cc, sp, cs, c: setattr(cs.out, 'vEgo', 5.),
  lambda cc, sp, cs, c: setattr(cs.out, 'vEgo', 22.2353),
  lambda cc, sp, cs, c: setattr(cs.out, 'vCruise', 80.),
  lambda cc, sp, cs, c: setattr(sp.leadOne, 'status', True),
  lambda cc, sp, cs, c: setattr(sp.leadTwo, 'status', True),
  lambda cc, sp, cs, c: setattr(sp, 'pcmCruiseCoastAllowed', False),
  lambda cc, sp, cs, c: setattr(sp, 'pcmCruiseCoastVersion', 2),
  lambda cc, sp, cs, c: setattr(sp, 'pcmCruiseCoastMonoTime', sp.pcmCruiseCoastMonoTime - 300_000_001),
  lambda cc, sp, cs, c: setattr(sp, 'pcmCruiseCoastMonoTime', sp.pcmCruiseCoastMonoTime + 1),
  lambda cc, sp, cs, c: setattr(sp, 'pcmCruiseCoastMonoTime', 0),
  lambda cc, sp, cs, c: setattr(c, 'launch_active', True),
]


@pytest.mark.parametrize('mutate', EXCLUSIONS)
def test_receiver_priority_exclusions_match_unmodified_path_on_first_frame(mutate):
  candidate, baseline = Rig(), Rig()
  expected = baseline.frame(allowed=False, mutate=mutate)[1]
  actual = candidate.frame(mutate=mutate)[1]
  assert actual == expected
  assert not candidate.ci.CC.cruise_coast.active


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('mutate', EXCLUSIONS[:15] + EXCLUSIONS[18:])
def test_priority_loss_immediately_leaves_active_coast(mode, mutate):
  rig = Rig(mode)
  for _ in range(30):
    rig.frame()
  assert rig.ci.CC.cruise_coast.active
  baseline = copy.deepcopy(rig)
  baseline.ci.CS.out = baseline.ci.CS.out.as_builder()
  baseline.ci.CC.cruise_coast.reset()
  for _ in range(10):
    actual = rig.frame(mutate=mutate)[3]
    expected = baseline.frame(allowed=False, mutate=mutate)[3]
    assert actual == expected
    assert not rig.ci.CC.cruise_coast.coasting


@pytest.mark.parametrize('car,interceptor', [(CAR.HONDA_CLARITY, False), (CAR.HONDA_CIVIC, True),
  (CAR.HONDA_CIVIC_BOSCH, False), (CAR.HONDA_RIDGELINE, False)])
def test_non_civic_pcm_paths_ignore_spurious_coast_metadata(car, interceptor):
  left, _ = interface(car, interceptor=interceptor, pcm_mode=0)
  right, _ = interface(car, interceptor=interceptor, pcm_mode=0)
  for tick in range(40):
    now = (100 + tick) * 10_000_000
    for ci in (left, right):
      ci.CS.out.canValid, ci.CS.out.vCruise = True, 72.
    cc = control(0.)
    cc.orientationNED = [0., -.07, 0.]
    sp = structs.CarControlSP(pcmCruiseCoastVersion=1, pcmCruiseCoastAllowed=True,
                             pcmCruiseCoastSetSpeed=20., pcmCruiseCoastMaxSpeed=22., pcmCruiseCoastMonoTime=now)
    assert left.apply(cc.as_reader(), sp, now)[1] == right.apply(cc.as_reader(), structs.CarControlSP(), now)[1]


@pytest.mark.parametrize('field', ['pcmCruiseCoastSetSpeed', 'pcmCruiseCoastMaxSpeed'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1., 0.])
def test_nonfinite_and_invalid_bounds_fail_closed(field, value):
  rig = Rig()
  rig.frame(mutate=lambda cc, sp, cs, c: setattr(sp, field, value))
  assert not rig.ci.CC.cruise_coast.active


@pytest.mark.parametrize('engine', [False, True])
def test_recorded_zero_accel_downhill_brake_fixture(engine):
  # Observed Civic frame: no lead, cruise plan/request=0, ~3.81-degree downhill,
  # 0.015 mph above set. Seeds reproduce the recorded 14-count brake command;
  # this is a command-path regression, not a simulation of changed vehicle motion.
  baseline, candidate = Rig(engine=engine), Rig(engine=engine)
  set_speed = 24.854848 * .44704
  for rig in (baseline, candidate):
    cs, c = rig.ci.CS.out, rig.ci.CC
    cs.vEgo, cs.vCruise, cs.aEgo = 24.869722 * .44704, set_speed * 3.6, -.024537327
    c.windfactor = c.windfactor_before_brake = 2.854537431
    c.nidec_pid.i = c.nidec_pid_factor = -.047451168
    c.nidec_brake_pid.i = c.brake_pid_factor = 1.66233
    c.brake_last = c.brake_steady = .140422158
    c.braking, c.apply_brake_last = True, 14
  _, _, _, original = baseline.frame(allowed=False, pitch=math.radians(-3.807813814), set_speed=set_speed)
  assert original['COMPUTER_BRAKE'] == 14 and original['BRAKE_LIGHTS'] == 1
  for _ in range(330):
    _, _, hud, brake = candidate.frame(pitch=math.radians(-3.807813814), set_speed=set_speed)
    if hud:
      assert hud['PCM_GAS'] == 0
    if brake and candidate.tick > 3:
      assert brake['COMPUTER_BRAKE'] == brake['BRAKE_LIGHTS'] == 0
  assert candidate.ci.CC.nidec_brake_pid.i == 1.66233


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('engine', [False, True])
def test_decoded_wire_entry_recovery_and_actual_wire_observer(mode, engine):
  rig = Rig(mode, engine=engine, learning=True)
  c, cs = rig.ci.CC, rig.ci.CS
  cs.car_gas_available, cs.car_gas, cs.engine_rpm = True, 30., 1500.
  c.average_factor = .035
  for _ in range(200):
    rig.frame(allowed=False, accel=.8, pitch=0.)
  assert c.pcm_gas_sent > 0
  entry_gas = c.pcm_gas_sent
  frozen = learned_state(c)
  gases = []
  for _ in range(320):
    before_average, before_wire = c.prior_gas_average, c.pcm_gas_sent
    out, _, hud, brake = rig.frame(accel=.8, pitch=0.)
    assert c.prior_gas_average == pytest.approx(before_average * (1 - c.average_factor) + before_wire * c.average_factor)
    assert learned_state(c) == frozen
    if hud:
      gases.append(hud['PCM_GAS'])
      assert out.gas == pytest.approx(hud['PCM_GAS'] / c.params.NIDEC_GAS_MAX)
    if brake:
      assert brake['COMPUTER_BRAKE'] == 0
  assert gases[0] <= entry_gas and gases[-1] == 0
  assert all(a >= b for a, b in zip(gases, gases[1:], strict=False))
  assert c.pcm_speed_sent == pytest.approx(0. if engine else 21., abs=.01)
  cs.out.vEgo = 19.5  # Below-set recovery, not a priority interruption.
  recovery = []
  for _ in range(310):
    _, _, hud, _ = rig.frame(accel=.8, pitch=0.)
    if hud:
      recovery.append(hud['PCM_GAS'])
      assert hud['PCM_GAS'] <= c.new_accel
  assert recovery[0] == 0 and recovery[-1] > 0
  assert not c.cruise_coast.active


@pytest.mark.parametrize('engine', [False, True])
def test_live_toggle_and_boundary_jitter_do_not_uncover_masked_gas(engine):
  rig = Rig(engine=engine)
  for t in range(150):
    if t % 10 == 0:
      rig.engine = not rig.engine
    rig.ci.CS.out.vEgo = 20.001 if t % 3 else 19.999
    _, _, hud, brake = rig.frame(pitch=0.)
    if hud:
      assert hud['PCM_GAS'] == 0
    if brake:
      assert brake['COMPUTER_BRAKE'] == 0
  for speed in (22.23519, 22.23521, 22.23519, 22.23521):
    rig.ci.CS.out.vEgo = speed
    rig.frame()
    assert rig.ci.CC.cruise_coast.coasting == (speed <= 22.2352)


def test_ecu_transition_timer_advances_while_explicit_coast_owns_brake_generation():
  rig = Rig(1)
  c = rig.ci.CC
  c.ecu_constraints.coast_frames = 20
  for tick in range(30):
    rig.frame()
    assert c.ecu_constraints.coast_frames == max(0, 19 - tick)
  assert c.cruise_coast.coasting
  baseline = copy.deepcopy(rig)
  baseline.ci.CC.cruise_coast.reset()
  counts = []
  for _ in range(100):
    _, _, _, brake = rig.frame(accel=-2.)
    _, _, _, expected = baseline.frame(allowed=False, accel=-2.)
    assert not c.cruise_coast.coasting
    if brake:
      assert brake == expected
      counts.append(brake['COMPUTER_BRAKE'])
  assert max(counts) > 0


@pytest.mark.parametrize('engine', [False, True])
def test_last_brake_frame_cannot_uncover_older_positive_hud_gas(engine):
  rig = Rig(engine=engine)
  c = rig.ci.CC
  c.pcm_gas_sent, c.pcm_speed_sent = 200, 25.
  c.apply_brake_last = 1
  c.brake_last, c.brake_steady, c.braking = .001, .001, True
  _, _, hud, brake = rig.frame(accel=.8, pitch=0.)
  assert brake['COMPUTER_BRAKE'] == 0
  assert hud['PCM_GAS'] == 0


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('engine', [False, True])
def test_positive_entry_finishes_fade_despite_lower_boundary_jitter(mode, engine):
  rig = Rig(mode, engine=engine)
  for _ in range(200):
    rig.frame(allowed=False, accel=.8, pitch=0.)
  assert rig.ci.CC.pcm_gas_sent > 100
  gases = []
  for tick in range(330):
    rig.ci.CS.out.vEgo = 20.001 if tick % 2 == 0 else 19.999
    _, _, hud, brake = rig.frame(accel=.8, pitch=0.)
    assert rig.ci.CC.cruise_coast.coasting
    if hud:
      gases.append(hud['PCM_GAS'])
    if brake:
      assert brake['COMPUTER_BRAKE'] == 0
  assert gases[-1] == 0
  assert all(a >= b for a, b in zip(gases, gases[1:], strict=False))


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('engine', [False, True])
def test_upper_boundary_denial_does_not_restore_hidden_gas(mode, engine):
  rig = Rig(mode, engine=engine)
  for _ in range(200):
    rig.frame(allowed=False, accel=.8, pitch=0.)
  for _ in range(320):
    rig.frame(accel=.8, pitch=0.)
  assert rig.ci.CC.pcm_gas_sent == 0
  for tick in range(40):
    rig.ci.CS.out.vEgo = 22.23521 if tick % 2 == 0 else 22.23519
    _, _, hud, _ = rig.frame(accel=.8, pitch=0.)
    assert rig.ci.CC.cruise_coast.coasting == (tick % 2 != 0)
    if hud:
      assert hud['PCM_GAS'] == 0
  # Leaving the allowed band restores the ordinary brake path on its first tick,
  # including its existing release/rise limits and wind credit, with no new delay.
  rig.ci.CS.out.vEgo = 22.3
  baseline = copy.deepcopy(rig)
  baseline.ci.CC.cruise_coast.reset()
  counts = []
  for _ in range(100):
    _, _, _, brake = rig.frame(accel=-2., pitch=0.)
    _, _, _, expected = baseline.frame(allowed=False, accel=-2., pitch=0.)
    assert not rig.ci.CC.cruise_coast.coasting
    if brake:
      assert brake == expected
      counts.append(brake['COMPUTER_BRAKE'])
  assert max(counts) > 0
