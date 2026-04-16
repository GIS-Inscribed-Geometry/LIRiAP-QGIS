import unittest
from unittest.mock import MagicMock, patch

from LIRiAP_pack import numba_bootstrap


class _Feedback:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def pushInfo(self, msg):
        self.infos.append(msg)

    def pushWarning(self, msg):
        self.warnings.append(msg)


class TestNumbaBootstrap(unittest.TestCase):
    def test_fast_path_when_numba_exists(self):
        feedback = _Feedback()
        with patch("LIRiAP_pack.numba_bootstrap.importlib.util.find_spec", return_value=object()):
            available, installed_now = numba_bootstrap.ensure_numba(feedback, attempt_install=True)
        self.assertTrue(available)
        self.assertFalse(installed_now)
        self.assertEqual(feedback.warnings, [])

    def test_no_install_when_disabled(self):
        feedback = _Feedback()
        with patch("LIRiAP_pack.numba_bootstrap.importlib.util.find_spec", return_value=None):
            available, installed_now = numba_bootstrap.ensure_numba(feedback, attempt_install=False)
        self.assertFalse(available)
        self.assertFalse(installed_now)

    def test_blocks_install_in_unsafe_context(self):
        feedback = _Feedback()
        with (
            patch("LIRiAP_pack.numba_bootstrap.importlib.util.find_spec", return_value=None),
            patch("LIRiAP_pack.numba_bootstrap._safe_auto_install_context", return_value=False),
        ):
            available, installed_now = numba_bootstrap.ensure_numba(feedback, attempt_install=True)
        self.assertFalse(available)
        self.assertFalse(installed_now)
        self.assertTrue(any("auto-install blocked" in w.lower() for w in feedback.warnings))

    def test_install_success_in_isolated_env(self):
        feedback = _Feedback()
        fake_proc = MagicMock(returncode=0, stderr="", stdout="")
        with (
            patch("LIRiAP_pack.numba_bootstrap.importlib.util.find_spec", side_effect=[None, object()]),
            patch("LIRiAP_pack.numba_bootstrap._safe_auto_install_context", return_value=True),
            patch("LIRiAP_pack.numba_bootstrap._in_isolated_python_env", return_value=True),
            patch("LIRiAP_pack.numba_bootstrap.subprocess.run", return_value=fake_proc) as run_mock,
        ):
            available, installed_now = numba_bootstrap.ensure_numba(feedback, attempt_install=True)
        self.assertTrue(available)
        self.assertTrue(installed_now)
        cmd = run_mock.call_args.args[0]
        self.assertIn("numba", cmd)
        self.assertNotIn("--user", cmd)

    def test_non_isolated_install_uses_user_site(self):
        feedback = _Feedback()
        fake_proc = MagicMock(returncode=0, stderr="", stdout="")
        with (
            patch("LIRiAP_pack.numba_bootstrap.importlib.util.find_spec", side_effect=[None, object()]),
            patch("LIRiAP_pack.numba_bootstrap._safe_auto_install_context", return_value=True),
            patch("LIRiAP_pack.numba_bootstrap._in_isolated_python_env", return_value=False),
            patch("LIRiAP_pack.numba_bootstrap.subprocess.run", return_value=fake_proc) as run_mock,
        ):
            available, installed_now = numba_bootstrap.ensure_numba(feedback, attempt_install=True)
        self.assertTrue(available)
        self.assertTrue(installed_now)
        cmd = run_mock.call_args.args[0]
        self.assertIn("--user", cmd)

    def test_install_failure_surfaces_warning(self):
        feedback = _Feedback()
        fake_proc = MagicMock(returncode=1, stderr="pip failure", stdout="")
        with (
            patch("LIRiAP_pack.numba_bootstrap.importlib.util.find_spec", return_value=None),
            patch("LIRiAP_pack.numba_bootstrap._safe_auto_install_context", return_value=True),
            patch("LIRiAP_pack.numba_bootstrap.subprocess.run", return_value=fake_proc),
        ):
            available, installed_now = numba_bootstrap.ensure_numba(feedback, attempt_install=True)
        self.assertFalse(available)
        self.assertFalse(installed_now)
        self.assertTrue(any("install failed" in w.lower() for w in feedback.warnings))


if __name__ == "__main__":
    unittest.main()
