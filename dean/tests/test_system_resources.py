import unittest
from unittest.mock import patch, MagicMock

from nlp.system_resources import (
    list_top_processes,
    ollama_call_is_safe,
    quit_process_gracefully,
)


def _proc(pid, name, username, rss_mb):
    proc = MagicMock()
    proc.info = {
        "pid": pid,
        "name": name,
        "username": username,
        "memory_info": MagicMock(rss=rss_mb * 1024 * 1024),
    }
    return proc


class TestOllamaCallIsSafe(unittest.TestCase):
    @patch("nlp.system_resources.swap_percent")
    @patch("nlp.system_resources.available_memory_mb")
    def test_blocks_when_free_memory_low(self, mock_free, mock_swap):
        mock_free.return_value = 400.0
        mock_swap.return_value = 10.0
        safe, reason = ollama_call_is_safe()
        self.assertFalse(safe)
        self.assertIn("free", reason)

    @patch("nlp.system_resources.swap_percent")
    @patch("nlp.system_resources.available_memory_mb")
    def test_blocks_when_swap_high(self, mock_free, mock_swap):
        mock_free.return_value = 5000.0
        mock_swap.return_value = 90.0
        safe, reason = ollama_call_is_safe()
        self.assertFalse(safe)
        self.assertIn("swap", reason)

    @patch("nlp.system_resources.swap_percent")
    @patch("nlp.system_resources.available_memory_mb")
    def test_safe_when_memory_healthy(self, mock_free, mock_swap):
        mock_free.return_value = 5000.0
        mock_swap.return_value = 10.0
        safe, reason = ollama_call_is_safe()
        self.assertTrue(safe)
        self.assertIsNone(reason)

    @patch("nlp.system_resources.swap_percent")
    @patch("nlp.system_resources.available_memory_mb")
    def test_fails_open_when_unreadable(self, mock_free, mock_swap):
        mock_free.return_value = None
        mock_swap.return_value = None
        safe, reason = ollama_call_is_safe()
        self.assertTrue(safe)


class TestListTopProcesses(unittest.TestCase):
    @patch("getpass.getuser", return_value="tester")
    @patch("os.getpid", return_value=111)
    @patch("psutil.process_iter")
    def test_excludes_self_ollama_and_other_users(self, mock_iter, mock_pid, mock_user):
        mock_iter.return_value = [
            _proc(111, "python", "tester", 500),  # self -- excluded
            _proc(222, "ollama", "tester", 2000),  # ollama -- excluded
            _proc(333, "Google Chrome", "tester", 800),
            _proc(444, "backgroundd", "root", 100),  # other user -- excluded
        ]
        rows = list_top_processes()
        pids = {r["pid"] for r in rows}
        self.assertEqual(pids, {333})
        self.assertEqual(rows[0]["mem_mb"], 800)

    @patch("getpass.getuser", return_value="tester")
    @patch("os.getpid", return_value=111)
    @patch("psutil.process_iter")
    def test_sorted_by_memory_descending(self, mock_iter, mock_pid, mock_user):
        mock_iter.return_value = [
            _proc(1, "A", "tester", 100),
            _proc(2, "B", "tester", 900),
            _proc(3, "C", "tester", 400),
        ]
        rows = list_top_processes()
        self.assertEqual([r["name"] for r in rows], ["B", "C", "A"])

    @patch("getpass.getuser", return_value="tester")
    @patch("os.getpid", return_value=111)
    @patch("psutil.process_iter")
    def test_respects_limit(self, mock_iter, mock_pid, mock_user):
        mock_iter.return_value = [_proc(i, f"proc{i}", "tester", i) for i in range(20)]
        rows = list_top_processes(limit=5)
        self.assertEqual(len(rows), 5)


class TestQuitProcessGracefully(unittest.TestCase):
    @patch("platform.system", return_value="Windows")
    def test_unsupported_platform_gives_clear_error(self, mock_system):
        ok, error = quit_process_gracefully(123, "Notepad")
        self.assertFalse(ok)
        self.assertIn("macOS", error)

    @patch("subprocess.run")
    @patch("platform.system", return_value="Darwin")
    def test_successful_quit_on_macos(self, mock_system, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        ok, error = quit_process_gracefully(123, "Google Chrome")
        self.assertTrue(ok)
        self.assertIsNone(error)
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "osascript")
        self.assertIn("Google Chrome", args[2])

    @patch("subprocess.run")
    @patch("platform.system", return_value="Darwin")
    def test_failed_quit_reports_error(self, mock_system, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="application not found")
        ok, error = quit_process_gracefully(123, "Nonexistent App")
        self.assertFalse(ok)
        self.assertIn("application not found", error)


if __name__ == "__main__":
    unittest.main()
