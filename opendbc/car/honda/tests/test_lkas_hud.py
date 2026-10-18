"""LKAS HUD flags and routing, without any vehicle I/O."""
from types import SimpleNamespace

import pytest

from opendbc.car.honda.hondacan import create_lkas_hud
from opendbc.car.honda.values import CAR, HondaFlags


class CapturePacker:
  def make_can_msg(self, name, bus, values):
    return name, bus, dict(values)


@pytest.mark.parametrize("ext_hud", [False, True])
@pytest.mark.parametrize("alt_radar", [False, True])
@pytest.mark.parametrize("open_long", [False, True])
@pytest.mark.parametrize("lat_active", [False, True])
def test_rdm_and_beep_flags_follow_hud_and_radar_type(ext_hud, alt_radar, open_long, lat_active):
  flags = HondaFlags.BOSCH
  if ext_hud:
    flags |= HondaFlags.BOSCH_EXT_HUD
  if alt_radar:
    flags |= HondaFlags.BOSCH_ALT_RADAR
  cp = SimpleNamespace(flags=flags, openpilotLongitudinalControl=open_long,
                       carFingerprint=CAR.ACURA_RDX_3G_MMR if alt_radar else CAR.HONDA_INSIGHT)
  messages = create_lkas_hud(
    CapturePacker(), 2, cp, None, lat_active, True, False, True, {}, False, False, SimpleNamespace(),
  )

  expected_names = ["LKAS_HUD_A", "LKAS_HUD_B"] if ext_hud and not open_long else ["LKAS_HUD"]
  assert [name for name, _, _ in messages] == expected_names
  expected = {
    "LKAS_READY": 1,
    "LKAS_STATE_CHANGE": 1,
    "STEERING_REQUIRED": True,
    "SOLID_LANES": lat_active,
    "DASHED_LANES": False,
    "BEEP": 0,
  }
  if not ext_hud or alt_radar:
    expected.update(RDM_OFF=1, LANE_ASSIST_BEEP_OFF=1)
  for _, bus, values in messages:
    assert bus == 2
    # Exact payload equality also protects normal extended HUDs: no new RDM or
    # beep fields there, and no unrelated lane/alert changes on ALT_RADAR.
    assert values == expected
