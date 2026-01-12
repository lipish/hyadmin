import os
from enum import Enum
from pathlib import Path


class LicMgrFlags:
    is_licensed: None | bool = None # already licensed previously
    is_sn_invalid: None | bool = None # whether serial number is pure ascii and of correct length
    is_license_invalid: None | bool = None # existing license is invalid
    is_collecting_failed: None | bool = None # failed to collect system data
    is_licensing_failed: None | bool = None # request for the new license failed
    licensing_error: str = ""

    class State(Enum):
        LICENSE_INVALID = "LICENSE_INVALID"
        SN_INVALID = "SN_INVALID"
        COLLECTING_FAILED = "COLLECTING_FAILED"
        LICENSING = "LICENSING"
        LICENSING_FAILED = "LICENSING_FAILED"
        LICENSED = "LICENSED"

    def get_state(self):
        if self.is_licensed:
            return LicMgrFlags.State.LICENSED
        if self.is_license_invalid:
            return LicMgrFlags.State.LICENSE_INVALID
        if self.is_sn_invalid:
            return LicMgrFlags.State.SN_INVALID
        if self.is_collecting_failed:
            return LicMgrFlags.State.COLLECTING_FAILED
        if self.is_licensing_failed:
            return LicMgrFlags.State.LICENSING_FAILED
        else:
            return LicMgrFlags.State.LICENSING

licmgr_flags = LicMgrFlags()
heyi_sn = ""


def check_license():
    assert (HEYI_LIC_DIR := os.getenv("HEYI_LIC_DIR", ".")), f"invalid {HEYI_LIC_DIR=}"
    global heyi_sn

    from heyi._ext import get_sn
    # def get_sn():
    #     if Path("heyi.lic").exists():
    #         return "0123456789abcdef"
    #     return ""

    heyi_sn = get_sn()
    licmgr_flags.is_licensed = heyi_sn != ""

    if licmgr_flags.is_licensed:
        return True
    
    if (Path(HEYI_LIC_DIR) / "heyi.lic").exists():
        licmgr_flags.is_license_invalid = True
        return False
    
