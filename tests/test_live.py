import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

os.environ.setdefault("TAPO_DATA", tempfile.mkdtemp(prefix="tapo-test-"))
os.environ.setdefault("TG_CHAT", "-1000000000001")
os.environ.setdefault("ARCHIVE_SOURCES", "101=Tapo C210,102=Tapo C200")
os.environ.setdefault("ARCHIVE_TOPICS", "Tapo C210=201,Tapo C200=202,Tapo C210 субтитры=203,Tapo C200 субтитры=204")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import archive  # noqa: E402
import backup  # noqa: E402
import live  # noqa: E402

TZ = timezone(timedelta(hours=3))


def clip_at(date, hour, minute, seconds=6):
    start = datetime.strptime(date, "%Y%m%d").replace(hour=hour, minute=minute, tzinfo=TZ)
    return mock.Mock(start=start, duration_seconds=seconds)


def make_mp4(path, seconds=6, size="320x240"):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=300:sample_rate=16000:duration={seconds}",
         "-c:v", "libx264", "-g", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )


class FakeDevice:
    def __init__(self, days, today="20261106", name="Tapo C200"):
        self.days, self.today, self.name = days, today, name
        self.searched = []

    async def camera_today(self):
        return self.today

    async def recording_dates(self, since, until):
        return sorted(set(self.days) | {self.today})

    async def search_day(self, date):
        self.searched.append(date)
        return self.days.get(date, [])


class PartTests(unittest.TestCase):
    def test_part_segments_shift_time_inside_split_clip(self):
        chapters = [(0.0, 100.0, clip_at("20260812", 10, 0)), (100.0, 100.0, clip_at("20260812", 11, 0))]
        first = live.part_segments(chapters, 0.0, 150.0)
        second = live.part_segments(chapters, 150.0, 50.0)
        self.assertEqual(len(first), 2)
        self.assertEqual(first[1]["pos"], 100.0)
        self.assertEqual(first[1]["end"], 150.0)
        self.assertEqual(first[1]["cam"], datetime(2026, 8, 12, 11, 0))
        self.assertEqual(first[1]["cam_end"], datetime(2026, 8, 12, 11, 0, 50))
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["pos"], 0.0)
        self.assertEqual(second[0]["cam"], datetime(2026, 8, 12, 11, 0, 50))
        self.assertEqual(second[0]["cam_end"], datetime(2026, 8, 12, 11, 1, 40))

    def test_caption_format_has_no_camera_name_and_notes(self):
        start = datetime(2026, 8, 12, 10, 0, 0)
        events = [{"pos": 0.0, "end": 60.0, "cam": start, "cam_end": start + timedelta(seconds=60)}]
        missing = [clip_at("20260812", 12, m) for m in range(12)]
        caption, rest = archive.build_caption(start, 60.0, events, live.part_note(1, 2, missing))
        lines = caption.split("\n")
        self.assertEqual(lines[0], "12.08.2026 10:00 - 10:01")
        self.assertEqual(lines[1], "1 мин 0 с видео 1 событие")
        self.assertEqual(lines[2], "часть 1/2")
        self.assertTrue(lines[3].startswith("не скачано: 12:00:00, 12:01:00"))
        self.assertTrue(lines[3].endswith("и ещё 2"))
        self.assertEqual(lines[4], "время в ролике - время камеры:")
        self.assertEqual(lines[5], "0:00 - 10:00:00")
        for bad in ("·", "—", "Tapo"):
            self.assertNotIn(bad, caption)
        self.assertEqual(rest, [])

    def test_keyframes_and_build_parts_split_on_resolution_change(self):
        work = Path(tempfile.mkdtemp(prefix="tapo-live-"))
        try:
            a, b, c = work / "a.mp4", work / "b.mp4", work / "c.mp4"
            make_mp4(a)
            make_mp4(b)
            make_mp4(c, size="640x360")
            self.assertEqual(live.keyframe_times(a)[:2], [0.0, 2.0])
            clips = [clip_at("20260812", 10, 0), clip_at("20260812", 10, 1), clip_at("20260812", 10, 2)]
            parts = live.build_parts(list(zip(clips, (a, b, c))), work)
            self.assertEqual(len(parts), 2)
            self.assertEqual(parts[0][3:], (320, 240))
            self.assertEqual(parts[1][3:], (640, 360))
            self.assertEqual(len(parts[0][1]), 2)
            self.assertEqual(parts[0][1][1]["cam"], datetime(2026, 8, 12, 10, 1))
            self.assertAlmostEqual(parts[0][2], 12, delta=0.5)
            self.assertEqual(parts[1][1][0]["pos"], 0.0)
        finally:
            shutil.rmtree(work)


class LoopTests(unittest.TestCase):
    def setUp(self):
        for path in (live.LIVE_WORK, archive.SPEECH_DIR):
            shutil.rmtree(path, ignore_errors=True)
        archive.WORK.mkdir(parents=True, exist_ok=True)
        archive.STATE_FILE.unlink(missing_ok=True)
        self.posts = []
        self.failing = set()

    def tearDown(self):
        for path in (live.LIVE_WORK, archive.SPEECH_DIR):
            shutil.rmtree(path, ignore_errors=True)
        archive.STATE_FILE.unlink(missing_ok=True)

    def make_run(self, state=None):
        if state:
            archive.save_json(archive.STATE_FILE, state)
        return archive.Run(mock.Mock(camera="", dry_run=False, date="", once=True), [])

    async def fake_fetch(self, s, dev, clip, path):
        if clip.start in self.failing:
            return False
        path.write_bytes(b"ts")
        return True

    def fake_convert(self, ts, mp4):
        make_mp4(mp4)

    async def fake_upload(self, client, chat, topic, path, caption, rest, duration, width, height, name):
        self.assertEqual(client, "poster")
        self.assertTrue(path.exists())
        self.posts.append((topic, name, caption))
        return [5000 + len(self.posts)]

    def run_pass(self, run, dev, cam_name="Tapo C200"):
        cam = mock.Mock(device_id="d1")
        cam.name = cam_name
        s = mock.Mock()
        s.device.return_value = dev
        with mock.patch.object(backup, "fetch_clip", self.fake_fetch), \
                mock.patch.object(backup, "clip_to_mp4", self.fake_convert), \
                mock.patch.object(archive, "upload", self.fake_upload), \
                mock.patch.object(archive, "ensure_topic", mock.AsyncMock(return_value=202)):
            return asyncio.run(live.camera_pass(run, s, cam, "poster", "chat"))

    def test_days_post_in_order_after_archive_and_skip_today(self):
        run = self.make_run({"items": {"7": {"camera": "Tapo C200", "name": "20260811_190356_tp00118.mp4", "messages": [1],
                                             "speech": []},
                                       "8": {"camera": "Tapo C210", "name": "20261001_190356_tp00118.mp4", "messages": [2],
                                             "speech": []}}})
        days = {"20260810": [clip_at("20260810", 9, 0)], "20260811": [clip_at("20260811", 22, 0)],
                "20260813": [clip_at("20260813", 11, 0), clip_at("20260813", 11, 1)],
                "20260812": [clip_at("20260812", 10, 0)], "20261106": [clip_at("20261106", 1, 0)]}
        dev = FakeDevice(days)
        self.assertTrue(self.run_pass(run, dev))
        self.assertEqual([p[1] for p in self.posts], ["20260812_100000_live.mp4", "20260813_110000_live.mp4"])
        self.assertEqual(dev.searched, ["20260812", "20260813"])
        self.assertEqual(self.posts[1][2].split("\n")[0], "13.08.2026 11:00 - 11:01")
        saved = archive.load_json(archive.STATE_FILE, {})
        self.assertTrue(saved["live"]["Tapo C200"]["20260813"]["done"])
        entry = saved["items"]["live_Tapo_C200_20260813_1"]
        self.assertEqual(entry["messages"], [5002])
        self.assertTrue(entry["live"])
        self.assertNotIn("speech", entry)
        names = sorted(p.name for p in archive.SPEECH_DIR.iterdir())
        self.assertEqual(names, [f"live_Tapo_C200_{d}_1.{x}" for d in ("20260812", "20260813")
                                 for x in ("json", "mp4", "opus")])
        self.assertEqual(list((live.LIVE_WORK / "Tapo_C200").iterdir()), [])
        meta = archive.load_json(archive.SPEECH_DIR / "live_Tapo_C200_20260813_1.json", {})
        self.assertEqual(meta["message"], 5002)
        self.assertAlmostEqual(meta["keyframes"][0], 0.0, delta=0.5)
        posted_again = list(self.posts)
        self.assertTrue(self.run_pass(self.make_run(), FakeDevice(days)))
        self.assertEqual(self.posts, posted_again)

    def test_failed_day_blocks_later_days_then_posts_with_missing_note(self):
        bad = clip_at("20260812", 10, 5)
        self.failing.add(bad.start)
        days = {"20260812": [clip_at("20260812", 10, 0), bad], "20260813": [clip_at("20260813", 11, 0)],
                "20261106": []}
        run = self.make_run()
        for attempt in range(live.DAY_ATTEMPTS - 1):
            self.assertFalse(self.run_pass(run, FakeDevice(days)))
            self.assertEqual(self.posts, [])
        self.assertTrue(self.run_pass(run, FakeDevice(days)))
        self.assertEqual(len(self.posts), 2)
        self.assertIn("не скачано: 10:05:00", self.posts[0][2])
        self.assertTrue(self.posts[0][1].startswith("20260812"))
        self.assertTrue(self.posts[1][1].startswith("20260813"))

    def test_day_with_nothing_downloaded_never_posts_and_later_gives_up(self):
        bad = clip_at("20260812", 10, 5)
        self.failing.add(bad.start)
        days = {"20260812": [bad], "20260813": [clip_at("20260813", 11, 0)], "20261106": []}
        run = self.make_run()
        self.assertFalse(self.run_pass(run, FakeDevice(days)))
        self.assertEqual(self.posts, [])
        run.state["live"]["Tapo C200"]["20260812"]["attempts"] = live.DAY_GIVE_UP - 1
        self.assertTrue(self.run_pass(run, FakeDevice(days)))
        self.assertEqual([p[1][:8] for p in self.posts], ["20260813"])
        self.assertEqual(run.state["live"]["Tapo C200"]["20260812"]["lost"], 1)

    def test_restart_keeps_posted_part_and_camera_error_is_contained(self):
        days = {"20260812": [clip_at("20260812", 10, 0)], "20261106": []}
        run = self.make_run()
        with mock.patch.object(live, "build_parts", side_effect=live.build_parts) as parts:
            self.assertTrue(self.run_pass(run, FakeDevice(days)))
        self.assertEqual(parts.call_count, 1)
        run.state["live"]["Tapo C200"]["20260812"].pop("done")
        dev = FakeDevice(days)
        self.assertTrue(self.run_pass(run, dev))
        self.assertEqual(dev.searched, ["20260812"])
        self.assertTrue(run.state["live"]["Tapo C200"]["20260812"]["done"])
        self.assertEqual(len(self.posts), 1)
        broken = FakeDevice(days)
        broken.search_day = mock.AsyncMock(side_effect=OSError("offline"))
        run.state["live"]["Tapo C200"].pop("20260812")
        self.assertFalse(self.run_pass(run, broken))

    def test_speech_loop_skips_lost_input_and_stops(self):
        run = self.make_run({"items": {"live_x_1": {"camera": "Tapo C200", "name": "20260812_100000_live.mp4",
                                                    "messages": [9], "live": True}}})
        stop = asyncio.Event()
        stop.set()
        with mock.patch.object(archive, "speech_item", side_effect=AssertionError("no input")):
            asyncio.run(live.speech_loop(run, "poster", "chat", None, None, stop))
        self.assertEqual(run.done["live_x_1"]["speech"], [])
        self.assertIn("missing", run.done["live_x_1"]["speech_skipped"])


if __name__ == "__main__":
    unittest.main()
