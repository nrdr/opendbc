"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from opendbc.car import structs, DT_CTRL
from opendbc.car.can_definitions import CanData
from opendbc.car.honda import hondacan
from opendbc.car.honda.values import CruiseButtons, HondaFlags
from opendbc.sunnypilot.car.honda.icbm_eligibility import civic_icbm_available
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP
from opendbc.sunnypilot.car.intelligent_cruise_button_management_interface_base import IntelligentCruiseButtonManagementInterfaceBase

ButtonType = structs.CarState.ButtonEvent.Type
SendButtonState = structs.IntelligentCruiseButtonManagement.SendButtonState

BUTTONS = {
  SendButtonState.increase: CruiseButtons.RES_ACCEL,
  SendButtonState.decrease: CruiseButtons.DECEL_SET,
}


class IntelligentCruiseButtonManagementInterface(IntelligentCruiseButtonManagementInterfaceBase):
  def __init__(self, CP, CP_SP):
    super().__init__(CP, CP_SP)

  def update(self, CC, CC_SP, CS, packer, frame, last_button_frame, CAN) -> list[CanData]:
    can_sends = []
    self.CC_SP = CC_SP
    self.ICBM = CC_SP.intelligentCruiseButtonManagement
    self.frame = frame
    self.last_button_frame = last_button_frame

    # ICBM changes factory set speed; it must never leak into OP-long modes.
    if self.CP.openpilotLongitudinalControl:
      return can_sends

    bus = None  # Preserve the existing Bosch route and command law.
    if not self.CP.flags & HondaFlags.BOSCH:
      if (not civic_icbm_available(self.CP, self.CP_SP) or not self.CP_SP.flags & HondaFlagsSP.STOCK_LONGITUDINAL or
          self.CP_SP.pcmCruiseSpeed or self.CP.passive or self.CP.dashcamOnly or not CC.enabled or
          CC.cruiseControl.cancel or CC.cruiseControl.resume or CC.cruiseControl.override or
          not CS.out.cruiseState.enabled or CS.out.cruiseState.standstill or CS.out.standstill or
          CS.out.gasPressed or CS.out.brakePressed or CS.cruise_buttons != 0 or CS.cruise_setting != 0):
        return can_sends
      # Host TX on the PT bus is not forwarded to the isolated stock camera.
      # Keep upstream button spam/counters; physical buttons continue forwarding.
      bus = CAN.camera

    if self.ICBM.sendButton != SendButtonState.none:
      send_button = BUTTONS[self.ICBM.sendButton]

      if (self.frame - self.last_button_frame) * DT_CTRL > 0.05:
        can_sends.append(hondacan.spam_buttons_command(packer, CAN, send_button, 0, CS.scm_ambient_light, self.CP, bus=bus))
        self.last_button_frame = self.frame

    return can_sends
