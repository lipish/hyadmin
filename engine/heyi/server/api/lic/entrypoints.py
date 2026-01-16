import asyncio
import os
from base64 import b64encode

import requests
from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from heyi.licmgr import heyi_sn, licmgr_flags

router = APIRouter()


class LicRequest(BaseModel):
    heyi_sn: str


@router.post("/request")
async def lic_request(body: LicRequest):
    from heyi._ext import gen_lic_info

    # def gen_lic_info(sn: str):
    #     if sn == "0123456789abcdef":
    #         return "licliclic" * 10
    #     return ""

    assert (HEYI_LIC_URL := os.getenv("HEYI_LIC_URL")), f"invalid {HEYI_LIC_URL=}"
    assert (HEYI_LIC_DIR := os.getenv("HEYI_LIC_DIR", ".")), f"invalid {HEYI_LIC_DIR=}"

    assert not licmgr_flags.is_licensed, "already licensed"
    assert not licmgr_flags.is_license_invalid, "license file exists but invalid"

    global heyi_sn
    heyi_sn = body.heyi_sn

    licmgr_flags.is_sn_invalid = not (len(heyi_sn) == 16 and heyi_sn.isascii())
    if licmgr_flags.is_sn_invalid:
        return False

    heyi_dat = b64encode(gen_lic_info(heyi_sn)).decode()

    if heyi_dat == "":
        licmgr_flags.is_collecting_failed = True
        return False
    licmgr_flags.is_collecting_failed = False

    data = {"serialNumber": heyi_sn, "blob": heyi_dat}

    retry_after = 2
    retry_count = 0
    last_error = ""

    while True:
        try:
            response = requests.post(HEYI_LIC_URL, json=data)
            ok = response.ok
            last_error = response.text if not ok else ""
        except Exception as e:
            response = None
            ok = False
            last_error = str(e)
            print(e)

        if ok:
            break

        if (retry_count := retry_count + 1) >= 3:
            with open(f"{HEYI_LIC_DIR}/heyi.dat", "w") as f:
                f.write(heyi_dat)
            licmgr_flags.is_licensing_failed = True
            licmgr_flags.licensing_error = last_error
            return {
                "status": "file_download",
                "filename": "heyi.dat",
                "download_url": "/lic/download-dat"
            }

        await asyncio.sleep(retry_after)

    assert response
    licmgr_flags.is_licensing_failed = False

    heyi_lic = response.json().get("license")
    with open(f"{HEYI_LIC_DIR}/heyi.lic", "w") as f:
        f.write(heyi_lic)
    return True


@router.get("/download-dat")
async def download_dat():
    """Endpoint to download the heyi.dat file"""
    HEYI_LIC_DIR = os.getenv("HEYI_LIC_DIR", ".")
    return FileResponse(
        path=f"{HEYI_LIC_DIR}/heyi.dat",
        media_type="application/octet-stream",
        filename="heyi.dat",
    )
