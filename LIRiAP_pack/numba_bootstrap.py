"""Shared helper for optional runtime Numba installation in LIRiAP wrappers."""

import importlib.util
import subprocess
import sys


def ensure_numba(feedback, attempt_install):
    """
    Returns (available, installed_now).
    """
    # Fast path: Numba is already available in the active Python environment.
    if importlib.util.find_spec("numba") is not None:
        return True, False

    if not attempt_install:
        return False, False

    # Optional self-bootstrap path requested by the user via algorithm parameter.
    feedback.pushInfo("Numba not found. Attempting installation via pip...")
    cmd = [sys.executable, "-m", "pip", "install", "numba"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if err:
            feedback.pushWarning(f"Numba install failed: {err[-500:]}")
        else:
            feedback.pushWarning("Numba install failed with unknown error.")
        return False, False

    available = importlib.util.find_spec("numba") is not None
    if available:
        feedback.pushInfo("Numba installed successfully.")
        return True, True

    feedback.pushWarning("Numba install command completed, but import check still failed.")
    return False, False
