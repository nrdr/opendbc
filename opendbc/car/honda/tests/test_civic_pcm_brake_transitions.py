"""Independent offline transition review of the Civic brake-feedback candidate.

Fixed-input checks establish controller behavior, not vehicle dynamics or safety.
The native-cadence test records held gas allowances; it does not validate the
physical propulsion-to-brake handoff. Assertions are not hidden behind xfails.
"""
import math
from hashlib import sha256

import pytest

from opendbc.car import structs
from opendbc.car.honda import civic_pcm
from opendbc.car.honda.civic_pcm_brake import CivicPCMBrakeFeedback
from opendbc.car.honda.tests.test_civic_pcm import control, interface
from opendbc.car.honda.tests.test_civic_pcm_speed_control import START, reference
from opendbc.car.honda.values import CAR, CarControllerParams


MAXIMUM = 255 / 256
LongCtrlState = structs.CarControl.Actuators.LongControlState
METADATA_FAILURES = ('stale', 'future', 'missing', 'nonfinite_reference', 'out_of_range_reference', 'zero_timestamp')

# Generated explicitly from the installed controller at
# 2cf0c7d0d27f60bdfeee8217b99c52c8dd66f6ba, using the shared host dependencies.
# Controller source SHA-256: 4416f307697b329f9419ce08e15e781beea0ab8fed90199298fed109d6a5b764.
# Never regenerate these from the candidate or HEAD. Tests need no Git history.
# The Bosch scenario uses lookup BP=[0., 2.], V=[0, 1600], fixed on its instance
# below because other Honda interfaces mutate the inherited class defaults.
INSTALLED_STOPPING_GOLDENS = {
  0.: ('91841061cb487a6259b8dca1154ab94d350e82326123353ccbfcbfb3113b3706', 1.3615926955649199e-05),
  1.99: ('a1f560ba46cd9992525aea2bd5c9c995a4e2340905ccd5157479b753247fadac', 0.),
  2.: ('a1f560ba46cd9992525aea2bd5c9c995a4e2340905ccd5157479b753247fadac', 0.),
}
INSTALLED_OUT_OF_SCOPE_GOLDENS = {
  (CAR.HONDA_CLARITY, False): ('b8b228fcc458284f67533ed274817cefaf8d92a470dc5dbfb1eaba161e6e1b32', 272),
  (CAR.HONDA_CLARITY, True): ('1605974f8e6333a18a9c801e3c1ad014bf12e03e06d7f140d42d64b570f0ad0d', 352),
  (CAR.HONDA_CIVIC, True): ('1605974f8e6333a18a9c801e3c1ad014bf12e03e06d7f140d42d64b570f0ad0d', 352),
  (CAR.HONDA_CIVIC_BOSCH, False): ('510906b98fcbcbbb53e7e074f5cf1fa610f72d804261cbdf69b0cfff9400f7b5', 400),
  (CAR.HONDA_RIDGELINE, False): ('b8b228fcc458284f67533ed274817cefaf8d92a470dc5dbfb1eaba161e6e1b32', 272),
}


def make_controller(*, wind=2.9863840730443507, car=CAR.HONDA_CIVIC, interceptor=False):
  ci, config = interface(car, interceptor=interceptor)
  ci.CS.out.canValid = True
  ci.CS.car_gas_available = True
  ci.CS.car_gas = 0.0
  ci.CC.windfactor = wind
  ci.CC.nidec_brake_pid.i = 2.179123328066013
  return ci, config


def tick(ci, index, *, requested=-.7, measured=.15, speed=23., pitch=-.05, target=22.,
         mode=LongCtrlState.pid, active=True, metadata=None):
  now = START + index * 10_000_000
  ci.CS.out.vEgo = ci.CS.out.vEgoRaw = float(speed)
  ci.CS.out.aEgo = float(measured)
  ci.CS.out.standstill = speed == 0.
  cc = control(requested)
  cc.longActive = active
  cc.actuators.longControlState = mode
  cc.orientationNED = [0., float(pitch), 0.]
  return ci.apply(cc.as_reader(), reference(target, now) if metadata is None else metadata, now)[1]


def feedback_update(pid, requested, measured, base, *, saturated=False):
  return pid.update(requested, measured, base, allowed=True, saturated=saturated, maximum=MAXIMUM, dt=.02)


def unavailable_metadata(reason, index):
  now = START + index * 10_000_000
  metadata = reference(22., now)
  if reason == 'stale':
    metadata.civicPcmReferenceMonoTime = now - civic_pcm.REFERENCE_MAX_AGE_NS - 1
  elif reason == 'future':
    metadata.civicPcmReferenceMonoTime = now + 1
  elif reason == 'missing':
    metadata = structs.CarControlSP()
  elif reason == 'nonfinite_reference':
    metadata.civicPcmReferenceSpeed = math.nan
  elif reason == 'out_of_range_reference':
    metadata.civicPcmReferenceSpeed = 101.
  elif reason == 'zero_timestamp':
    metadata.civicPcmReferenceMonoTime = 0
  else:
    raise ValueError(reason)
  return metadata


def assert_no_gas_brake_overlap(ci, messages):
  brake_packets = [int.from_bytes(payload[:2], 'big') >> 6 for address, payload, _ in messages if address == 0x1FA]
  if brake_packets:
    assert brake_packets == [ci.CC.apply_brake_last]
  if ci.CC.apply_brake_last > 0:
    assert ci.CC.new_accel == 0
    assert ci.CC.pcm_gas_sent == 0, 'Nonzero brake overlaps the last transmitted PCM gas allowance'
  for address, payload, _ in messages:
    if address == 0x30C and ci.CC.apply_brake_last > 0:
      assert payload[2] == 0 and payload[:2] == b'\x00\x00'


def add_packet_tick(digest, index, messages):
  # Frame each tick and packet explicitly, preserving packet order, CAN IDs,
  # buses, payload lengths and every payload byte, including empty send ticks.
  digest.update(index.to_bytes(4, 'big'))
  digest.update(len(messages).to_bytes(2, 'big'))
  for address, payload, bus in messages:
    digest.update(address.to_bytes(4, 'big'))
    digest.update(bus.to_bytes(1, 'big'))
    digest.update(len(payload).to_bytes(1, 'big'))
    digest.update(payload)


def test_positive_integral_unwinds_when_proportional_term_clips_output_to_zero():
  pid = CivicPCMBrakeFeedback()
  for _ in range(50):
    feedback_update(pid, -.7, .15, 0.)
  assert pid.integral > .15

  for _ in range(100):
    assert feedback_update(pid, -.7, -2., 0.) == 0.
  assert pid.integral == pytest.approx(0., abs=1e-12), "Lower saturation must allow positive demand to unwind"

  # Matching the request after sustained excess deceleration must not revive
  # a hidden old brake demand.
  assert feedback_update(pid, -.7, -.7, 0.) == 0.


def test_negative_integral_unwinds_when_proportional_term_clips_output_to_maximum():
  pid = CivicPCMBrakeFeedback()
  for _ in range(100):
    feedback_update(pid, -.7, -1., .8)
  before = pid.integral
  assert before < -.1
  feedback_update(pid, -.7, 2., .8)
  assert pid.integral > before, "Upper saturation must allow a negative trim to unwind"


def test_loss_of_coasting_eligibility_does_not_restore_excess_feedforward():
  ci, _ = make_controller(wind=.1)
  for index in range(300):
    tick(ci, index, requested=-.3, measured=-.5, pitch=0.)
  assert ci.CC.civic_brake_feedback.integral < -.01
  previous = ci.CC.apply_brake_last
  assert previous == 0

  # CAR_GAS is an ECU measurement, not a driver-gas override. The request
  # and excess measured deceleration are unchanged across this threshold.
  ci.CS.car_gas = 1.01
  tick(ci, 300, requested=-.3, measured=-.5, pitch=0.)
  assert ci.CC.apply_brake_last <= previous, "Dropping eligibility must not add brake during over-deceleration"


def test_crossing_two_meters_per_second_does_not_discard_needed_braking():
  ci, _ = make_controller()
  for index in range(300):
    tick(ci, index, measured=0., speed=2.01, pitch=0.)
  previous = ci.CC.apply_brake_last
  assert previous > 150
  for index in range(300, 320):
    tick(ci, index, measured=0., speed=2., pitch=0.)
  assert ci.CC.apply_brake_last >= previous - 1, "A 0.01 m/s change must not discard braking while error is unchanged"


def test_crossing_two_meters_per_second_does_not_restore_trimmed_excess_braking():
  ci, _ = make_controller()
  for index in range(300):
    tick(ci, index, measured=-1., speed=2.01, pitch=0.)
  assert ci.CC.civic_brake_feedback.integral < -.1
  previous = ci.CC.apply_brake_last
  assert previous == 0
  tick(ci, 300, measured=-1., speed=2., pitch=0.)
  assert ci.CC.apply_brake_last <= previous, "Low-speed handoff must not add braking while already over-decelerating"


def test_positive_plan_below_reference_does_not_restore_trimmed_grade_feedforward():
  ci, _ = make_controller(wind=.1)
  for index in range(300):
    tick(ci, index, requested=-.3, measured=-.8)
  assert ci.CC.civic_brake_feedback.integral < -.05
  previous = ci.CC.apply_brake_last
  assert previous == 0
  # A positive request below the planner speed disables downhill_overspeed.
  # The grade FF still demands brake, and the legacy gain multiplies it.
  tick(ci, 300, requested=.1, measured=-.8, target=24.)
  assert ci.CC.apply_brake_last <= previous, "Increasing the plan must not reapply excess grade braking"


def test_flat_positive_plan_clears_feedback_and_preserves_factory_release_slew():
  ci, _ = make_controller()
  for index in range(300):
    tick(ci, index)
  previous = ci.CC.apply_brake_last
  assert previous > 0
  for index in range(300, 320):
    tick(ci, index, requested=.5, measured=1., pitch=0., target=23.)
    assert ci.CC.civic_brake_feedback.integral == ci.CC.civic_brake_feedback.feedback == 0.
    if index % 2 == 0:
      assert ci.CC.apply_brake_last == max(previous - 32, 0)
      previous = ci.CC.apply_brake_last
  assert ci.CC.apply_brake_last == 0


def test_stopping_feedback_preserves_existing_stopping_rise_envelope():
  ci, _ = make_controller()
  previous = ci.CC.apply_brake_last
  for index in range(20):
    tick(ci, index, mode=LongCtrlState.stopping)
  # Existing Nidec stopping code permits +0.00042 normalized brake per 20 ms.
  # One wire unit of tolerance allows packing/rounding at the envelope edge.
  maximum_step = 10 * .00042 * ci.CC.params.NIDEC_BRAKE_MAX + 1
  assert ci.CC.apply_brake_last - previous <= maximum_step, "Feedback is packed after the stopping limiter"


@pytest.mark.parametrize('speed', [0., 1.99, 2.])
def test_inactive_stopping_retains_fractional_ramp(speed):
  ci, _ = make_controller()
  signature, first_effort = INSTALLED_STOPPING_GOLDENS[speed]
  digest = sha256()
  packet_count = 0
  for index in range(200):
    messages = tick(ci, index, speed=speed, pitch=0., mode=LongCtrlState.stopping)
    add_packet_tick(digest, index, messages)
    packet_count += len(messages)
    expected_effort = first_effort + (index // 2) * .00042
    assert ci.CC.last_applied_brake == pytest.approx(expected_effort, rel=0., abs=1e-12)
    assert ci.CC.civic_brake_feedback.integral == ci.CC.civic_brake_feedback.feedback == 0.
  assert packet_count == 340
  assert digest.hexdigest() == signature
  assert .04 < ci.CC.last_applied_brake <= 100 * .00042
  assert ci.CC.apply_brake_last == 10


def test_inactive_stopping_handoff_keeps_sub_packet_accumulation():
  ci, _ = make_controller()
  ci.CC.last_applied_brake = .01
  ci.CC.apply_brake_last = int(.01 * ci.CC.params.NIDEC_BRAKE_MAX)
  ci.CC.brake_last = .2
  ci.CC.civic_brake_feedback.integral = ci.CC.civic_brake_feedback.feedback = .2
  for index in range(20):
    tick(ci, index, speed=2., pitch=0., mode=LongCtrlState.stopping)
  assert ci.CC.last_applied_brake == pytest.approx(.01 + 10 * .00042)
  assert ci.CC.apply_brake_last == int(ci.CC.last_applied_brake * ci.CC.params.NIDEC_BRAKE_MAX)
  assert ci.CC.civic_brake_feedback.integral == ci.CC.civic_brake_feedback.feedback == 0.


@pytest.mark.parametrize('base', [0., .2, .8, MAXIMUM])
def test_rebased_integral_remains_inside_current_feedforward_bounds(base):
  pid = CivicPCMBrakeFeedback()
  for old_base, measured in [(0., .5), (.8, -2.), (.2, .4), (MAXIMUM, -1.)]:
    for _ in range(50):
      feedback_update(pid, -.7, measured, old_base)
    output = feedback_update(pid, -.7, .2, base)
    assert -base <= pid.integral <= MAXIMUM - base
    assert 0. <= output <= MAXIMUM


def test_lower_saturation_does_not_accumulate_more_negative_trim():
  pid = CivicPCMBrakeFeedback()
  for _ in range(500):
    assert feedback_update(pid, -.7, -5., .1) == 0.
  assert pid.integral == 0.


def test_upper_saturation_does_not_accumulate_more_positive_demand():
  pid = CivicPCMBrakeFeedback()
  for _ in range(500):
    assert feedback_update(pid, -.7, .5, MAXIMUM, saturated=True) == MAXIMUM
  assert pid.integral == 0.


@pytest.mark.parametrize('missing_metadata', [False, True])
@pytest.mark.parametrize('reason', ['disengage', 'driver_gas', 'driver_brake', 'aeb', 'invalid_can'])
def test_authority_loss_clears_feedback_by_next_brake_packet(reason, missing_metadata):
  ci, _ = make_controller()
  for index in range(100):
    tick(ci, index)
  assert ci.CC.civic_brake_feedback.integral > .1

  active = reason != 'disengage'
  if reason == 'driver_gas':
    ci.CS.out.gasPressed = True
  elif reason == 'driver_brake':
    ci.CS.out.brakePressed = True
  elif reason == 'aeb':
    ci.CS.out.stockAeb = True
  elif reason == 'invalid_can':
    ci.CS.out.canValid = False
  metadata = unavailable_metadata('missing', 100) if missing_metadata else None
  tick(ci, 100, active=active, metadata=metadata)
  assert ci.CC.civic_brake_feedback.integral == ci.CC.civic_brake_feedback.feedback == 0.
  assert not ci.CC.civic_brake_feedback.active
  assert ci.CC.civic_pcm_speed_control  # Missing metadata must not select the old law.


@pytest.mark.parametrize('reason', METADATA_FAILURES)
@pytest.mark.parametrize('requested', [-.7, 0.])
def test_metadata_loss_preserves_required_braking_under_live_authority(reason, requested):
  ci, _ = make_controller()
  for index in range(300):
    tick(ci, index, requested=requested)
  previous = ci.CC.apply_brake_last
  integral = ci.CC.civic_brake_feedback.integral
  assert previous > 0 and integral > 0.
  for index in range(300, 350):
    messages = tick(ci, index, requested=requested, metadata=unavailable_metadata(reason, index))
    assert ci.CC.civic_brake_feedback.active
    assert ci.CC.civic_brake_feedback.integral >= integral - 1e-12
    assert ci.CC.apply_brake_last >= previous - 1, 'Metadata loss alone must not release needed braking'
    assert_no_gas_brake_overlap(ci, messages)
  assert ci.CC.civic_pcm_speed_control


@pytest.mark.parametrize('reason', METADATA_FAILURES)
def test_metadata_loss_preserves_negative_trim_against_excess_grade_feedforward(reason):
  ci, _ = make_controller(wind=.1)
  for index in range(300):
    tick(ci, index, requested=-.3, measured=-.8)
  integral = ci.CC.civic_brake_feedback.integral
  assert integral < -.05 and ci.CC.apply_brake_last == 0
  for index in range(300, 350):
    messages = tick(ci, index, requested=-.3, measured=-.8, metadata=unavailable_metadata(reason, index))
    assert ci.CC.civic_brake_feedback.active
    assert ci.CC.civic_brake_feedback.integral <= integral + 1e-12
    assert ci.CC.apply_brake_last == 0, 'Metadata loss must not restore the old multiplied grade FF'
    assert ci.CC.new_accel == ci.CC.pcm_gas_sent == 0
    assert_no_gas_brake_overlap(ci, messages)


@pytest.mark.parametrize('reason', METADATA_FAILURES)
def test_positive_request_without_metadata_releases_friction_and_keeps_gas_zero(reason):
  ci, _ = make_controller()
  for index in range(300):
    tick(ci, index)
  previous = ci.CC.apply_brake_last
  assert previous > 0
  for index in range(300, 330):
    messages = tick(ci, index, requested=.1, metadata=unavailable_metadata(reason, index))
    assert ci.CC.civic_brake_feedback.integral == ci.CC.civic_brake_feedback.feedback == 0.
    assert not ci.CC.civic_brake_feedback.active
    if index % 2 == 0:
      assert ci.CC.apply_brake_last == max(previous - 32, 0)
      previous = ci.CC.apply_brake_last
    assert ci.CC.new_accel == ci.CC.pcm_gas_sent == 0
    assert_no_gas_brake_overlap(ci, messages)
  assert ci.CC.apply_brake_last == 0


@pytest.mark.parametrize('reason', METADATA_FAILURES)
def test_positive_request_without_metadata_cannot_restore_trimmed_grade_braking(reason):
  ci, _ = make_controller(wind=.1)
  for index in range(300):
    tick(ci, index, requested=-.3, measured=-.8)
  assert ci.CC.apply_brake_last == 0 and ci.CC.civic_brake_feedback.integral < -.05
  for index in range(300, 330):
    messages = tick(ci, index, requested=.1, measured=-.8, metadata=unavailable_metadata(reason, index))
    assert ci.CC.apply_brake_last == 0
    assert ci.CC.civic_brake_feedback.integral == ci.CC.civic_brake_feedback.feedback == 0.
    assert ci.CC.new_accel == ci.CC.pcm_gas_sent == 0
    assert_no_gas_brake_overlap(ci, messages)


@pytest.mark.parametrize('reason', METADATA_FAILURES)
def test_missing_metadata_does_not_end_rolling_low_speed_brake_ownership(reason):
  ci, _ = make_controller()
  for index in range(100):
    tick(ci, index, speed=2.01, measured=0., pitch=0.)
  previous = ci.CC.apply_brake_last
  for index in range(100, 120):
    messages = tick(ci, index, speed=1.5, measured=0., pitch=0., metadata=unavailable_metadata(reason, index))
    assert ci.CC.civic_brake_feedback.active
    assert ci.CC.apply_brake_last >= previous - 1
    assert_no_gas_brake_overlap(ci, messages)


@pytest.mark.parametrize('reason', METADATA_FAILURES)
def test_reference_recovery_cannot_revive_released_brake_trim(reason):
  ci, _ = make_controller()
  for index in range(100):
    tick(ci, index)
  for index in range(100, 130):
    tick(ci, index, requested=.5, pitch=0., metadata=unavailable_metadata(reason, index))
  assert ci.CC.apply_brake_last == 0
  for index in range(130, 160):
    messages = tick(ci, index, requested=.5, measured=.15, pitch=0., target=24.)
    assert ci.CC.apply_brake_last == 0
    assert ci.CC.civic_brake_feedback.integral == ci.CC.civic_brake_feedback.feedback == 0.
    assert_no_gas_brake_overlap(ci, messages)
  assert ci.CC.pcm_gas_sent > 0


@pytest.mark.parametrize('missing_metadata', [False, True])
@pytest.mark.parametrize('phase', range(10))
def test_native_handoff_records_held_allowance_until_next_scheduled_hud(phase, missing_metadata):
  """Characterize native cadence, not a guarantee of physical torque exclusion."""
  ci, _ = make_controller()
  first = 100 + phase
  for index in range(first):
    tick(ci, index, requested=.5, measured=.15, pitch=0., target=24.)
  assert ci.CC.pcm_gas_sent > 0 and ci.CC.apply_brake_last == 0
  prior_allowance = ci.CC.pcm_gas_sent
  first_hud_tick = ((first + 9) // 10) * 10
  held_overlap_ticks = []
  saw_brake = False
  for index in range(first, first + 20):
    metadata = unavailable_metadata('missing', index) if missing_metadata else None
    messages = tick(ci, index, metadata=metadata)
    brake_packets = [int.from_bytes(payload[:2], 'big') >> 6 for address, payload, _ in messages if address == 0x1FA]
    hud_packets = [payload for address, payload, _ in messages if address == 0x30C]
    assert len(brake_packets) == int(index % 2 == 0)
    assert len(hud_packets) == int(index % 10 == 0)
    for brake in brake_packets:
      assert brake == ci.CC.apply_brake_last
      if brake > 0:
        saw_brake = True
        assert ci.CC.new_accel == 0  # Internal permission is removed immediately.
        if ci.CC.pcm_gas_sent > 0:
          held_overlap_ticks.append(index)
    for payload in hud_packets:
      assert payload[2] == 0
      if ci.CC.apply_brake_last > 0:
        assert payload[:2] == b'\x00\x00'
    if index < first_hud_tick:
      assert ci.CC.pcm_gas_sent == prior_allowance
    else:
      assert ci.CC.pcm_gas_sent == 0
  assert saw_brake
  # Do not disguise nonzero held allowance as "zero wire overlap". Under the
  # retained 50/10 Hz schedule it persists for 0..80 ms after the first brake,
  # measured at controller-step boundaries, with counts [0,4,4,3,3,2,2,1,1,0].
  # The ECU response and intra-batch transmission order are not modeled here.
  assert held_overlap_ticks == [index for index in range(first, first_hud_tick) if index % 2 == 0]


def test_initially_missing_metadata_does_not_suppress_live_negative_brake_request():
  ci, _ = make_controller()
  saw_brake = False
  for index in range(100):
    messages = tick(ci, index, metadata=unavailable_metadata('missing', index))
    saw_brake |= ci.CC.apply_brake_last > 0
    assert_no_gas_brake_overlap(ci, messages)
  assert saw_brake, 'A valid live negative CarControl must not depend on receiving the first SP packet'


def test_positive_request_without_first_metadata_cannot_apply_grade_braking_or_gas():
  ci, _ = make_controller(wind=.1)
  for index in range(100):
    messages = tick(ci, index, requested=.1, measured=.7, metadata=unavailable_metadata('missing', index))
    assert ci.CC.apply_brake_last == 0, 'The first missing SP packet must not select old grade braking for a positive request'
    assert ci.CC.new_accel == ci.CC.pcm_gas_sent == 0
    assert_no_gas_brake_overlap(ci, messages)


def test_braking_to_stop_and_launch_cannot_carry_feedback_into_propulsion():
  ci, _ = make_controller()
  for index in range(100):
    tick(ci, index)
  for index in range(100, 200):
    tick(ci, index, speed=0., measured=0., target=0., mode=LongCtrlState.stopping)
    assert ci.CC.civic_brake_feedback.integral == ci.CC.civic_brake_feedback.feedback == 0.
  messages = []
  for index in range(200, 240):
    messages.extend(tick(ci, index, requested=.8, measured=0., speed=0., target=0., mode=LongCtrlState.starting))
    assert ci.CC.civic_brake_feedback.integral == ci.CC.civic_brake_feedback.feedback == 0.
  assert ci.CC.launch_active
  assert ci.CC.apply_brake_last == 0
  assert any(address == 0x30C and payload[2] > 0 for address, payload, _ in messages)


def test_startup_speed_control_freezes_preexisting_integrals_without_metadata():
  ci, _ = make_controller(wind=.1)
  assert ci.CC.civic_pcm_speed_control
  ci.CC.nidec_pid.i = -.5
  legacy_integral = ci.CC.nidec_brake_pid.i
  for index in range(200):
    metadata = unavailable_metadata('missing', index) if index < 100 else None
    tick(ci, index, requested=-1., pitch=0., metadata=metadata)
    assert ci.CC.civic_pcm_speed_control
    assert ci.CC.nidec_brake_pid.i == legacy_integral
    assert ci.CC.nidec_pid.i == ci.CC.nidec_pid_factor == 0.
    assert ci.CC.accel == pytest.approx(-1.)


@pytest.mark.parametrize('car,interceptor', [(CAR.HONDA_CLARITY, False), (CAR.HONDA_CLARITY, True),
                                           (CAR.HONDA_CIVIC, True), (CAR.HONDA_CIVIC_BOSCH, False),
                                           (CAR.HONDA_RIDGELINE, False)])
def test_non_civic_pcm_paths_do_not_enter_feedback_across_modes(car, interceptor):
  ci, _ = make_controller(car=car, interceptor=interceptor)

  def unexpected(*args, **kwargs):
    pytest.fail('Out-of-scope vehicle entered Civic brake feedback')

  ci.CC.civic_brake_feedback.update = unexpected
  for index in range(120):
    mode = LongCtrlState.pid if index < 40 else LongCtrlState.stopping if index < 80 else LongCtrlState.starting
    tick(ci, index, mode=mode, requested=-.7 if index < 80 else .8, speed=23. if index < 80 else 0.)
  assert not ci.CC.civic_pcm_speed_control


@pytest.mark.parametrize('car,interceptor', [(CAR.HONDA_CLARITY, False), (CAR.HONDA_CLARITY, True),
                                           (CAR.HONDA_CIVIC, True), (CAR.HONDA_CIVIC_BOSCH, False),
                                           (CAR.HONDA_RIDGELINE, False)])
def test_out_of_scope_packets_match_installed_controller_golden(car, interceptor):
  signature, expected_packets = INSTALLED_OUT_OF_SCOPE_GOLDENS[car, interceptor]
  actual_signature, packet_count, last_brake = out_of_scope_packet_signature(car, interceptor)
  assert packet_count == expected_packets
  assert actual_signature == signature, (car, interceptor)
  assert last_brake == 0


def out_of_scope_packet_signature(car, interceptor):
  ci, _ = make_controller(car=car, interceptor=interceptor)
  if car == CAR.HONDA_CIVIC_BOSCH:
    # Declare the original installed-controller golden's calibration inputs.
    # Instance fields isolate this test without altering or repairing globals.
    ci.CC.params.BOSCH_GAS_LOOKUP_BP = [0., 2.]
    ci.CC.params.BOSCH_GAS_LOOKUP_V = [0, 1600]
  digest = sha256()
  packet_count = 0
  for index in range(160):
    requested = 1. if index < 40 else -.3 if index < 80 else -2. if index < 120 else .8
    mode = LongCtrlState.stopping if 80 <= index < 120 else LongCtrlState.pid
    messages = tick(ci, index, requested=requested, measured=.3 if index < 80 else -.6, mode=mode)
    add_packet_tick(digest, index, messages)
    packet_count += len(messages)
  return digest.hexdigest(), packet_count, ci.CC.apply_brake_last


def test_bosch_golden_is_independent_of_other_platform_class_lookup_mutations(monkeypatch):
  # This combination reproduced the broad-suite digest 7a16bdc953ba... under
  # BOTH the installed and candidate controllers before scenario isolation.
  monkeypatch.setattr(CarControllerParams, 'BOSCH_GAS_LOOKUP_BP', [-.2, 2.])
  monkeypatch.setattr(CarControllerParams, 'BOSCH_GAS_LOOKUP_V', [0, 2200])
  signature, packet_count, last_brake = out_of_scope_packet_signature(CAR.HONDA_CIVIC_BOSCH, False)
  expected_signature, expected_count = INSTALLED_OUT_OF_SCOPE_GOLDENS[CAR.HONDA_CIVIC_BOSCH, False]
  assert (signature, packet_count, last_brake) == (expected_signature, expected_count, 0)
  assert CarControllerParams.BOSCH_GAS_LOOKUP_BP == [-.2, 2.]
  assert CarControllerParams.BOSCH_GAS_LOOKUP_V == [0, 2200]
