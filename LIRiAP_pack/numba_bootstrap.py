"""
Shared helper for optional runtime Numba installation in LIRiAP wrappers.

Provides safe auto-installation of the Numba JIT compiler when requested
via algorithm parameters. Only attempts installation in safe contexts:
- Isolated Python environments (venv, conda)
- Writable user-site directories

Functions
=========
ensure_numba(feedback, attempt_install) -> (available, installed_now)
  Checks for existing Numba and optionally installs if missing.

_in_isolated_python_env() -> bool
  Detects venv/conda environments.

_user_site_writable() -> bool
  Checks if user-site directory is writable.

_safe_auto_install_context() -> bool
  Returns True if auto-install is safe.

See Also
========
*_algorithm.py: Algorithm wrappers that use ensure_numba
"""

import importlib.util
import os
import site
import subprocess
import sys


def _in_isolated_python_env():
    """Return True for venv/conda-like interpreter environments."""
    if hasattr(sys, "real_prefix"):
        return True
    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return True
    if os.environ.get("CONDA_PREFIX"):
        return True
    return False


def _user_site_writable():
    """Best-effort check for a writable user-site target."""
    try:
        user_site = site.getusersitepackages()
    except Exception:
        return False
    if not user_site:
        return False
    try:
        os.makedirs(user_site, exist_ok=True)
        return os.access(user_site, os.W_OK)
    except Exception:
        return False


def _safe_auto_install_context():
    """
    Allow pip self-bootstrap only in relatively safe contexts:
    - isolated env (venv/conda), or
    - user site is writable (so install can stay user-scoped).
    """
    return _in_isolated_python_env() or _user_site_writable()


def ensure_numba(feedback, attempt_install):
    """
    Check for Numba availability and optionally install it.

    Parameters
    ----------
    feedback : QgisFeedback
        QGIS feedback object for user messages.
    attempt_install : bool
        If True, attempt to install Numba if not available.

    Returns
    -------
    tuple
        (available: bool, installed_now: bool)
    """
    # Fast path: Numba is already available in the active Python environment.
    if importlib.util.find_spec("numba") is not None:
        return True, False

    if not attempt_install:
        return False, False

    # Optional self-bootstrap path requested by the user via algorithm parameter.
    if not _safe_auto_install_context():
        feedback.pushWarning(
            "Numba auto-install blocked: environment is not isolated and user-site "
            "is not writable. Install numba manually or run inside a venv/conda env."
        )
        return False, False

    feedback.pushInfo("Numba not found. Attempting installation via pip...")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--upgrade-strategy",
        "only-if-needed",
    ]
    if not _in_isolated_python_env():
        cmd.append("--user")
    cmd.append("numba")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=300
        )
    except subprocess.TimeoutExpired:
        feedback.pushWarning("Numba install timed out after 300s.")
        return False, False
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

    feedback.pushWarning(
        "Numba install command completed, but import check still failed."
    )
    return False, False
