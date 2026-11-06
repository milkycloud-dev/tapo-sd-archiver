import asyncio
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("TAPO_DATA", tempfile.mkdtemp(prefix="tapo-test-"))
os.environ.setdefault("TG_CHAT", "-1000000000001")
os.environ.setdefault("ARCHIVE_SOURCES", "101=Tapo C210,102=Tapo C200")
os.environ.setdefault("ARCHIVE_TOPICS", "Tapo C210=201,Tapo C200=202,Tapo C210 субтитры=203,Tapo C200 субтитры=204")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import archive  # noqa: E402
import backup  # noqa: E402
import control  # noqa: E402
import live  # noqa: E402
import panel  # noqa: E402
from ontapo.errors import AuthError, MFARequired  # noqa: E402


class FakeSession:
    instances = []

    def __init__(self, token="tok", refresh_token="ref", dead=False, refresh_ok=True, email="a@b.c"):
        self.token, self.refresh_token, self.dead, self.refresh_ok, self.email = token, refresh_token, dead, refresh_ok, email
        self.closed = self.code_sent = 0
        self.app_server = ""
        self.submitted = []
        self.reject_code = False
        FakeSession.instances.append(self)

    def to_dict(self):
        return {"token": self.token, "refreshToken": self.refresh_token, "email": self.email}

    async def cameras(self):
        if self.dead:
            raise AuthError("expired")
        return []

    async def refresh(self):
        if not self.refresh_ok:
            raise AuthError("refresh failed")
        self.token, self.refresh_token, self.dead = "tok2", "ref2", False

    async def aclose(self):
        self.closed += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def send_mfa_code(self, mfa_type=2):
        self.code_sent += 1

    async def submit_mfa(self, code, mfa_type=2):
        if self.reject_code:
            raise AuthError("bad code")
        self.submitted.append(code)
        self.token, self.refresh_token = "newtok", "newref"


class Base(unittest.TestCase):
    def setUp(self):
        FakeSession.instances.clear()
        for path in (control.SETTINGS_FILE, control.STATUS_FILE, control.CONTROL_FILE, control.CREDENTIALS_FILE,
                     backup.SESSION_FILE):
            path.unlink(missing_ok=True)
        for source in panel.LOG_SOURCES:
            control.log_path(source).unlink(missing_ok=True)


class ControlTests(Base):
    def test_settings_validate_ranges_and_types(self):
        self.assertEqual(control.load_settings()["interval_min"], 60)
        saved = control.save_settings({"interval_min": "30", "queue_gb": "1,5", "paused": True})
        self.assertEqual(saved["interval_min"], 30)
        self.assertIsInstance(saved["interval_min"], int)
        self.assertEqual(saved["queue_gb"], 1.5)
        self.assertTrue(control.load_settings()["paused"])
        for bad in ({"interval_min": "1"}, {"memory_gate_pct": "99"}, {"retry_min": "abc"}, {"min_free_gb": "1"}):
            with self.assertRaises(ValueError):
                control.save_settings(bad)
        self.assertEqual(control.load_settings()["interval_min"], 30)

    def test_credentials_private_and_override_env(self):
        with mock.patch.dict(os.environ, {"TAPO_EMAIL": "env@x.y", "TAPO_PASSWORD": "envpass"}):
            self.assertEqual(control.load_credentials(), ("env@x.y", "envpass"))
            control.save_credentials(" new@x.y ", "newpass")
            self.assertEqual(control.load_credentials(), ("new@x.y", "newpass"))
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(control.CREDENTIALS_FILE.stat().st_mode), 0o600)
        self.assertEqual(control.mask_email("milky@example.com"), "mi***@example.com")
        self.assertEqual(control.mask_email(""), "не задан")

    def test_log_tail_and_rotation(self):
        with mock.patch.dict(os.environ, {"TAPO_LOG_SOURCE": "live"}), mock.patch("builtins.print"):
            backup.log("first line")
        self.assertIn("first line", control.tail_log("live"))
        with mock.patch.object(control, "LOG_LIMIT", 10):
            control.append_log("live", "second line")
        self.assertEqual(control.tail_log("live").strip(), "second line")
        self.assertTrue(control.log_path("live").with_suffix(".log.1").exists())
        control.log_path("live").with_suffix(".log.1").unlink()
        self.assertEqual(control.tail_log("absent"), "")

    def test_control_counters(self):
        self.assertEqual(control.read_control()["wake"], 0)
        control.bump_control("wake")
        control.bump_control("wake")
        control.bump_control("restart")
        self.assertEqual(control.read_control(), {"wake": 2, "restart": 1})


class SavedSessionTests(Base):
    def open(self, session):
        with mock.patch.object(backup.OnTapo, "from_dict", return_value=session):
            return asyncio.run(backup.open_saved_session())

    def test_no_session_requires_login_and_never_attempts_password(self):
        with mock.patch.object(backup.OnTapo, "login", side_effect=AssertionError("password login attempted")):
            with self.assertRaises(backup.LoginRequired):
                asyncio.run(backup.open_saved_session())

    def test_valid_session_is_not_rewritten(self):
        control.write_json(backup.SESSION_FILE, {"token": "tok", "refreshToken": "ref"})
        before = backup.SESSION_FILE.read_text()
        session = FakeSession()
        self.assertIs(self.open(session), session)
        self.assertEqual(backup.SESSION_FILE.read_text(), before)

    def test_expired_token_refreshes_and_saves(self):
        control.write_json(backup.SESSION_FILE, {"token": "tok", "refreshToken": "ref"})
        session = FakeSession(dead=True)
        self.assertIs(self.open(session), session)
        self.assertEqual(control.read_json(backup.SESSION_FILE, {})["refreshToken"], "ref2")
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(backup.SESSION_FILE.stat().st_mode), 0o600)

    def test_dead_refresh_requires_login_without_password_attempt(self):
        control.write_json(backup.SESSION_FILE, {"token": "tok", "refreshToken": "ref"})
        session = FakeSession(dead=True, refresh_ok=False)
        with mock.patch.object(backup.OnTapo, "login", side_effect=AssertionError("password login attempted")):
            with self.assertRaises(backup.LoginRequired):
                self.open(session)
        self.assertEqual(session.closed, 1)


class LoginFlowTests(Base):
    def setUp(self):
        super().setUp()
        control.save_credentials("a@b.c", "secret-pass")

    def mfa_login(self):
        session = FakeSession(token="", refresh_token="")
        return session, mock.AsyncMock(side_effect=MFARequired("p1", [2], session=session))

    def test_code_flow_saves_session_and_keeps_secrets_out_of_logs(self):
        session, login = self.mfa_login()
        flow = panel.LoginFlow()
        with mock.patch.dict(os.environ, {"TAPO_LOG_SOURCE": "panel"}), mock.patch("builtins.print"), \
                mock.patch.object(panel.OnTapo, "login", login):
            self.assertEqual(asyncio.run(flow.start()), "code_sent")
            self.assertEqual(session.code_sent, 1)
            self.assertTrue(flow.waiting)
            self.assertFalse(backup.SESSION_FILE.exists())
            asyncio.run(flow.submit(" 123456 "))
        self.assertEqual(session.submitted, ["123456"])
        self.assertEqual(control.read_json(backup.SESSION_FILE, {})["token"], "newtok")
        self.assertFalse(flow.waiting)
        self.assertEqual(session.closed, 1)
        logged = control.tail_log("panel")
        self.assertIn("new session saved", logged)
        self.assertNotIn("secret-pass", logged)
        self.assertNotIn("123456", logged)
        self.assertNotIn("newtok", logged)

    def test_double_click_sends_one_code_then_resend_after_pause(self):
        session, login = self.mfa_login()
        flow = panel.LoginFlow()

        async def scenario():
            with mock.patch.object(panel.OnTapo, "login", login):
                await flow.start()
                await flow.start()
                self.assertEqual(login.await_count, 1)
                self.assertEqual(session.code_sent, 1)
                flow.sent_at -= panel.RESEND_SEC + 1
                await flow.start()
                self.assertEqual(login.await_count, 2)
                self.assertEqual(session.closed, 1)

        with mock.patch("builtins.print"):
            asyncio.run(scenario())

    def test_wrong_code_keeps_waiting_and_expired_code_is_refused(self):
        session, login = self.mfa_login()
        flow = panel.LoginFlow()

        async def scenario():
            with mock.patch.object(panel.OnTapo, "login", login):
                await flow.start()
            session.reject_code = True
            with self.assertRaises(AuthError):
                await flow.submit("000000")
            self.assertTrue(flow.waiting)
            self.assertFalse(backup.SESSION_FILE.exists())
            flow.sent_at -= panel.CODE_TTL + 1
            with self.assertRaises(TimeoutError):
                await flow.submit("111111")
            self.assertIsNone(flow.pending)

        with mock.patch("builtins.print"):
            asyncio.run(scenario())

    def test_login_without_code_and_missing_credentials(self):
        ready = FakeSession(token="direct", refresh_token="r")
        flow = panel.LoginFlow()
        with mock.patch.object(panel.OnTapo, "login", mock.AsyncMock(return_value=ready)), mock.patch("builtins.print"):
            self.assertEqual(asyncio.run(flow.start()), "ok")
        self.assertEqual(control.read_json(backup.SESSION_FILE, {})["token"], "direct")
        control.CREDENTIALS_FILE.unlink()
        with mock.patch.dict(os.environ, {"TAPO_EMAIL": "", "TAPO_PASSWORD": ""}):
            with self.assertRaises(AuthError):
                asyncio.run(panel.LoginFlow().start())

    def test_keep_alive_refreshes_and_reports_failure(self):
        self.assertFalse(asyncio.run(panel.keep_alive()))
        control.write_json(backup.SESSION_FILE, {"token": "tok", "refreshToken": "ref"})
        good = FakeSession()
        with mock.patch.object(panel.OnTapo, "from_dict", return_value=good), mock.patch("builtins.print"):
            self.assertTrue(asyncio.run(panel.keep_alive()))
        self.assertEqual(control.read_json(backup.SESSION_FILE, {})["refreshToken"], "ref2")
        bad = FakeSession(refresh_token="ref2", refresh_ok=False)
        with mock.patch.object(panel.OnTapo, "from_dict", return_value=bad), mock.patch("builtins.print"):
            self.assertFalse(asyncio.run(panel.keep_alive()))
        self.assertEqual(bad.closed, 1)


class RenderTests(Base):
    def snap(self, **status):
        return {"status": {"updated": time.time(), "state": "running", "login": "ok", "cameras": {}, **status},
                "settings": control.load_settings(), "free_gb": 12.3, "queue_gb": 1.2, "now": time.time(),
                "archive": {"Tapo C200": {"posts": 119, "speech": 100, "pending": 3, "last": "20260812_100000_live.mp4"}},
                "archive_age": 1e9}

    def fields(self, embed):
        return {f.name: f.value for f in embed.fields}

    def test_healthy_view_shows_camera_progress_and_sd(self):
        probe = {"valid": True, "checked": time.time(), "cameras": [{"name": "Tapo C200", "sd": "normal 29 GB"}]}
        snap = self.snap(cameras={"Tapo C200": {"date": "20260813", "clips": "3/10", "pending": 80}})
        embed, needed = panel.render(snap, probe, False)
        self.assertFalse(needed)
        self.assertEqual(embed.color.value, panel.GREEN)
        fields = self.fields(embed)
        self.assertIn("Действует", fields["Вход Tapo"])
        self.assertIn("работает", fields["Живой поток"])
        for piece in ("Постов 119", "Субтитры ждут: 3", "12.08.2026", "13.08.2026, клипы 3/10", "Дней осталось: 80", "normal 29 GB"):
            self.assertIn(piece, fields["Tapo C200"])

    def test_login_required_is_red_and_code_prompt_wins(self):
        embed, needed = panel.render(self.snap(), {"valid": False, "checked": time.time()}, False)
        self.assertTrue(needed)
        self.assertEqual(embed.color.value, panel.RED)
        self.assertIn("Нужен вход", self.fields(embed)["Вход Tapo"])
        embed, _ = panel.render(self.snap(login="required"), {}, True)
        self.assertIn("Ввести код", self.fields(embed)["Вход Tapo"])

    def test_stale_status_means_not_running_and_archive_is_named(self):
        snap = self.snap(updated=time.time() - panel.STALE_SEC - 1)
        snap["archive_age"] = 10
        embed, needed = panel.render(snap, {}, False)
        self.assertFalse(needed)
        self.assertEqual(embed.color.value, panel.YELLOW)
        text = self.fields(embed)["Живой поток"]
        self.assertIn("не запущен", text)
        self.assertIn("работает архив", text)

    def test_password_never_rendered(self):
        control.save_credentials("milky@example.com", "secret-pass")
        embed, _ = panel.render(self.snap(), {}, False)
        text = str(embed.to_dict())
        self.assertNotIn("secret-pass", text)
        self.assertNotIn("milky@example.com", text)
        self.assertIn("mi***@example.com", text)

    def test_archive_summary_counts(self):
        archive.save_json(panel.ARCHIVE_STATE, {"items": {
            "1": {"camera": "A", "name": "20260101_000000_tp1.mp4", "messages": [1], "speech": [2]},
            "2": {"camera": "A", "name": "20260102_000000_tp2.mp4", "messages": [3], "speech": []},
            "3": {"camera": "A", "name": "20260103_000000_live.mp4", "messages": [4], "live": True},
            "4": {"camera": "B", "name": "x", "skipped": "err"},
        }})
        try:
            self.assertEqual(panel.archive_summary(), {"A": {"posts": 3, "speech": 1, "pending": 1,
                                                             "last": "20260103_000000_live.mp4"}})
        finally:
            panel.ARCHIVE_STATE.unlink()

    def test_feed_keeps_only_important_lines(self):
        keep = ["2026 Tapo C200 20260812: posted", "2026 speech x failed (1/5): E", "2026 login required: no saved session"]
        skip = ["2026   Tapo C200 20260812: 10/25 clips", "2026   uploaded x.mp4 at 5 MB/s", "2026 speech x: 3 phrases"]
        self.assertTrue(all(panel.FEED_RE.search(line) for line in keep))
        self.assertFalse(any(panel.FEED_RE.search(line) for line in skip))


class LiveControlTests(Base):
    def make_run(self):
        return archive.Run(mock.Mock(camera="", dry_run=False, date="", once=True), [])

    def test_pause_and_room_block_and_memory_gate_applies(self):
        run = self.make_run()
        self.assertEqual(live.blocked_reason(run), "")
        control.save_settings({"memory_gate_pct": 45})
        live.blocked_reason(run)
        self.assertEqual(archive.MEMORY_GATE, 0.45)
        control.save_settings({"paused": True})
        self.assertEqual(live.blocked_reason(run), "paused")
        control.save_settings({"paused": False})
        with mock.patch.object(archive, "free_gb", return_value=1.0):
            self.assertEqual(live.blocked_reason(run), "waiting")
        archive.MEMORY_GATE = 0.6

    def test_paused_stream_resumes_when_unpaused(self):
        run = self.make_run()
        control.save_settings({"paused": True})

        async def scenario():
            task = asyncio.create_task(live.wait_for_room(run))
            await asyncio.sleep(0.05)
            self.assertFalse(task.done())
            self.assertEqual(control.read_status()["state"], "paused")
            control.save_settings({"paused": False})
            await asyncio.wait_for(task, 2)
            self.assertEqual(control.read_status()["state"], "running")

        with mock.patch.object(live, "ROOM_POLL", 0.01), mock.patch("builtins.print"):
            asyncio.run(scenario())

    def test_login_required_waits_without_login_and_new_session_wakes(self):
        run = self.make_run()
        run.args.once = False
        calls = []

        async def fake_open():
            calls.append(1)
            if len(calls) == 1:
                raise backup.LoginRequired("expired")
            raise asyncio.CancelledError

        async def scenario():
            with mock.patch.object(backup, "open_saved_session", fake_open), mock.patch("builtins.print"):
                task = asyncio.create_task(live.main_loop(run, None, None, asyncio.Event()))
                await asyncio.sleep(0.05)
                self.assertEqual(len(calls), 1)
                status = control.read_status()
                self.assertEqual(status["state"], "login_required")
                self.assertEqual(status["login"], "required")
                control.write_json(backup.SESSION_FILE, {"token": "new"})
                os.utime(backup.SESSION_FILE, (1, 1))
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 8)
                self.assertEqual(len(calls), 2)

        asyncio.run(scenario())

    def test_failed_camera_retries_while_other_camera_is_busy(self):
        run = self.make_run()
        run.args.once = False
        cams = [mock.Mock(app_server="", device_id=str(i)) for i in range(2)]
        cams[0].name, cams[1].name = "Busy", "Failing"
        calls = {"Busy": 0, "Failing": 0}
        retried = asyncio.Event()

        async def fake_pass(run_, s, cam, poster, post_chat):
            calls[cam.name] += 1
            if cam.name == "Busy":
                await retried.wait()
                return True
            if calls[cam.name] == 2:
                retried.set()
                return True
            return False

        session = FakeSession()
        session.cameras = mock.AsyncMock(return_value=cams)

        async def fake_idle(delay):
            await asyncio.sleep(0.01)

        async def scenario():
            with mock.patch.object(backup, "open_saved_session", mock.AsyncMock(return_value=session)), \
                    mock.patch.object(live, "camera_pass", fake_pass), \
                    mock.patch.object(live, "idle", fake_idle), \
                    mock.patch("builtins.print"):
                task = asyncio.create_task(live.main_loop(run, None, None, asyncio.Event()))
                await asyncio.wait_for(retried.wait(), 3)
                self.assertEqual(calls["Failing"], 2)
                self.assertEqual(calls["Busy"], 1)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        asyncio.run(scenario())

    def test_once_mode_finishes_and_sets_stop(self):
        run = self.make_run()
        cam = mock.Mock(app_server="")
        cam.name = "Solo"
        session = FakeSession()
        session.cameras = mock.AsyncMock(return_value=[cam])
        stop = asyncio.Event()
        camera_pass = mock.AsyncMock(return_value=True)
        with mock.patch.object(backup, "open_saved_session", mock.AsyncMock(return_value=session)), \
                mock.patch.object(live, "camera_pass", camera_pass), mock.patch("builtins.print"):
            asyncio.run(live.main_loop(run, None, None, stop))
        self.assertEqual(camera_pass.await_count, 1)
        self.assertTrue(stop.is_set())

    def test_idle_returns_on_wake(self):
        async def scenario():
            task = asyncio.create_task(live.idle(60))
            await asyncio.sleep(0.05)
            control.bump_control("wake")
            await asyncio.wait_for(task, 8)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
