from __future__ import annotations

import os


HOSTED_FREE_PROFILE = "hosted_free"
LOCAL_FULL_PROFILE = "local_full"


def app_profile() -> str:
    profile = (os.getenv("APP_PROFILE") or HOSTED_FREE_PROFILE).strip().lower()
    if profile in {HOSTED_FREE_PROFILE, LOCAL_FULL_PROFILE}:
        return profile
    return HOSTED_FREE_PROFILE


def profile_default(*, hosted_free: str, local_full: str) -> str:
    if app_profile() == LOCAL_FULL_PROFILE:
        return local_full
    return hosted_free
