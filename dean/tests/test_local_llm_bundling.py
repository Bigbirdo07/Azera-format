import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

from core.privacy_guard import assert_loopback_url, PrivacyGuardError
from nlp.local_model_manager import BundledOllamaManager, LLMRuntimeStatus


class TestLocalLLMBundling(unittest.TestCase):

    def test_privacy_guard_allowed_ports(self):
        # 11434 and 11438-11442 should pass
        assert_loopback_url("http://127.0.0.1:11434")
        assert_loopback_url("http://127.0.0.1:11438")
        assert_loopback_url("http://127.0.0.1:11442")
        assert_loopback_url("http://localhost:11438")

    def test_privacy_guard_blocked_endpoints(self):
        # Remote IP, HTTPS, and outside port range should raise errors
        with self.assertRaises(PrivacyGuardError):
            assert_loopback_url("https://127.0.0.1:11434")  # HTTPS blocked
        with self.assertRaises(PrivacyGuardError):
            assert_loopback_url("http://google.com:11434")   # Non-loopback host blocked
        with self.assertRaises(PrivacyGuardError):
            assert_loopback_url("http://127.0.0.1:11443")  # Outside port range blocked

    @patch("platform.system")
    @patch("platform.machine")
    def test_unsupported_platform_gives_clear_error(self, mock_machine, mock_system):
        mock_system.return_value = "FreeBSD"
        mock_machine.return_value = "x86_64"

        manager = BundledOllamaManager()
        bin_path, err = manager._get_binary_path()
        self.assertIsNone(bin_path)
        self.assertIn("Unsupported operating system", err)

    @patch("platform.system")
    @patch("platform.machine")
    def test_unsupported_architecture_gives_clear_error(self, mock_machine, mock_system):
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "mips"

        manager = BundledOllamaManager()
        bin_path, err = manager._get_binary_path()
        self.assertIsNone(bin_path)
        self.assertIn("Unsupported macOS CPU architecture", err)

    @patch("subprocess.Popen")
    @patch("pathlib.Path.exists")
    @patch("nlp.local_model_manager.BundledOllamaManager._get_binary_path")
    def test_duplicate_start_prevention(self, mock_get_bin, mock_exists, mock_popen):
        mock_get_bin.return_value = (Path("/fake/bin/ollama-darwin-arm64"), None)
        mock_exists.return_value = True

        manager = BundledOllamaManager()
        # Mock active running status
        manager.process = MagicMock()
        manager.process.poll.return_value = None
        manager.status = LLMRuntimeStatus(
            mode="bundled_ollama",
            endpoint=manager.url,
            port=manager.port,
            model_name="llama3.2:3b",
            server_running=True,
            model_available=True,
            fallback_used=False
        )

        res = manager.start("llama3.2:3b")
        self.assertTrue(res)
        mock_popen.assert_not_called()

    def test_safe_stop_behavior_no_process(self):
        manager = BundledOllamaManager()
        manager.process = None
        # Should execute cleanly without raising NoneType exceptions
        manager.stop()
        self.assertEqual(manager.status.server_running, False)
