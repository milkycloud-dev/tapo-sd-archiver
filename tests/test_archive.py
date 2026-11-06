import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

os.environ["TAPO_DATA"] = tempfile.mkdtemp(prefix="tapo-test-")
os.environ["TG_CHAT"] = "-1000000000001"
os.environ["ARCHIVE_SOURCES"] = "101=Tapo C210,102=Tapo C200"
os.environ["ARCHIVE_TOPICS"] = "Tapo C210=201,Tapo C200=202,Tapo C210 субтитры=203,Tapo C200 субтитры=204"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import archive  # noqa: E402
import speech  # noqa: E402
from archive import (  # noqa: E402
    CAPTION_LIMIT, MESSAGE_LIMIT, MARGIN, build_caption, build_segments, build_speech_caption, cam_at,
    merge_events, plural, speech_spans, split_text,
)

START = datetime(2024, 7, 27, 13, 42, 45)


def event(pos, end, cam, cam_end):
    return {"pos": pos, "end": end, "cam": cam, "cam_end": cam_end}


def make_video(path, seconds=12, audio=True):
    args = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=15:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=8000:duration={seconds}", "-c:a", "pcm_alaw"]
    args += ["-c:v", "libx264", "-g", "30", "-pix_fmt", "yuv420p", "-f", "mov", str(path)]
    subprocess.run(args, check=True)


class CaptionTests(unittest.TestCase):
    def test_main_caption_format(self):
        events = [
            event(0, 300, START, START + timedelta(seconds=300)),
            event(362, 600, START + timedelta(minutes=34), START + timedelta(minutes=38)),
        ]
        caption, rest = build_caption(START, 3020, events)
        lines = caption.split("\n")
        self.assertEqual(lines[0], "27.07.2024 13:42 - 14:20")
        self.assertEqual(lines[1], "50 мин 20 с видео 2 события")
        self.assertEqual(lines[2], "время в ролике - время камеры:")
        self.assertEqual(lines[3], "0:00 - 13:42:45")
        self.assertEqual(lines[4], "6:02 - 14:16:45")
        for bad in ("·", "—", "–", "Tapo"):
            self.assertNotIn(bad, caption)
        self.assertEqual(rest, [])

    def test_multi_day_span_and_labels(self):
        late = datetime(2024, 7, 27, 23, 50, 0)
        events = [event(0, 100, late, late + timedelta(seconds=100)),
                  event(100, 200, late + timedelta(hours=1), late + timedelta(hours=1, seconds=100))]
        caption, _ = build_caption(late, 200, events)
        self.assertTrue(caption.startswith("27.07.2024 23:50 - 28.07.2024 00:51\n"))
        self.assertIn("1:40 - 28.07 00:50:00", caption)

    def test_plural(self):
        cases = {1: "событие", 2: "события", 4: "события", 5: "событий", 11: "событий", 12: "событий",
                 21: "событие", 22: "события", 111: "событий", 101: "событие"}
        for n, want in cases.items():
            self.assertEqual(plural(n, "событие", "события", "событий"), want, n)

    def test_split_text_limits_and_order(self):
        lines = [f"{i}:00 - 12:00:{i % 60:02} " + "слово " * (i % 7) for i in range(1500)]
        caption, rest = split_text("заголовок", lines)
        self.assertLessEqual(len(caption), CAPTION_LIMIT)
        for chunk in rest:
            self.assertLessEqual(len(chunk), MESSAGE_LIMIT)
        rebuilt = caption.split("\n")[1:] + [line for chunk in rest for line in chunk.split("\n")]
        self.assertEqual(rebuilt, lines)

    def test_split_text_long_line(self):
        caption, rest = split_text("h", ["x" * 9000])
        self.assertEqual(caption, "h")
        self.assertEqual("".join(rest), "x" * 9000)
        self.assertTrue(all(len(c) <= MESSAGE_LIMIT for c in rest))


class TimelineTests(unittest.TestCase):
    def test_segments_with_jump_and_noise(self):
        pts = np.arange(0, 40, 2.0)
        offsets = [0.0] * 10 + [600.0] * 10
        offsets[3] = 5000.0
        offsets[4] = None
        segments, quality = build_segments(offsets, pts, START, 40.0)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["pos"], 0.0)
        self.assertEqual(segments[0]["cam"], START)
        self.assertEqual(segments[1]["pos"], 20.0)
        self.assertEqual(segments[1]["cam"], START + timedelta(seconds=620))
        self.assertEqual(segments[1]["end"], 40.0)
        self.assertGreater(quality, 0.85)

    def test_unreadable_start_falls_back_to_filename(self):
        pts = np.arange(0, 20, 2.0)
        offsets = [None] * 3 + [300.0] * 7
        segments, _ = build_segments(offsets, pts, START, 20.0)
        self.assertEqual(segments[0]["pos"], 0.0)
        self.assertEqual(segments[0]["cam"], START)
        self.assertEqual(segments[1]["cam"], START + timedelta(seconds=306))

    def test_nothing_readable(self):
        segments, quality = build_segments([None] * 5, np.arange(0, 10, 2.0), START, 10.0)
        self.assertEqual(len(segments), 1)
        self.assertEqual(quality, 0.0)
        self.assertEqual(segments[0]["cam_end"], START + timedelta(seconds=10))

    def test_merge_events(self):
        a = event(0, 10, START, START + timedelta(seconds=10))
        b = event(10, 20, START + timedelta(seconds=60), START + timedelta(seconds=70))
        c = event(20, 30, START + timedelta(seconds=600), START + timedelta(seconds=610))
        merged = merge_events([a, b, c])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["end"], 20)
        self.assertEqual(merged[0]["cam_end"], START + timedelta(seconds=70))

    def test_cam_at(self):
        segs = [event(0, 10, START, START), event(10, 20, START + timedelta(hours=1), START)]
        self.assertEqual(cam_at(segs, 5), START + timedelta(seconds=5))
        self.assertEqual(cam_at(segs, 12), START + timedelta(hours=1, seconds=2))


class SpeechTests(unittest.TestCase):
    def test_spans_snap_pad_merge(self):
        keyframes = [float(k) for k in range(0, 100, 2)]
        phrases = [(10.5, 12.0, "а"), (15.0, 16.0, "б"), (60.3, 61.0, "в")]
        spans = speech_spans(phrases, 70.0, keyframes)
        self.assertEqual(spans, [[8.0, 18.0], [58.0, 63.0]])
        self.assertEqual(speech_spans([(0.5, 1.0, "x")], 2.0, keyframes), [[0.0, 2.0]])

    def test_speech_caption_offsets(self):
        segs = [event(0, 100, START, START + timedelta(seconds=100))]
        phrases = [(10.5, 12.0, "привет"), (60.3, 61.0, "пока")]
        spans = [[8.0, 18.0], [58.0, 63.0]]
        caption, rest = build_speech_caption(START, 100, segs, segs, phrases, spans, [10.0, 5.0], 15.0, "LINK")
        self.assertIn("полное видео: LINK", caption)
        self.assertIn("0:02 - 13:42:55 привет", caption)
        self.assertIn("0:12 - 13:43:45 пока", caption)
        self.assertNotIn("·", caption)


class OcrTests(unittest.TestCase):
    def test_vectorized_reader_matches_reference(self):
        rng = np.random.default_rng(1)
        geom = {"xs": [5 + 20 * i for i in range(18)], "gw": 12, "y0": 4, "y1": 20}
        templates = np.stack([archive.norm(rng.standard_normal((16, 12)).astype(np.float32))
                              for _ in archive.GLYPHS])
        frames = (rng.random((7, 30, 380)) * 255).astype(np.uint8)
        fast = archive.read_digits(frames, geom, templates, batch=3)
        for f, row in zip(frames, fast):
            ref = [int(np.einsum("sij,dij->sd", archive.glyph_shifts(w, geom), templates).max(0).argmax())
                   for w in archive.glyph_windows(f.astype(np.float32), geom)]
            self.assertEqual(list(row), ref)

    def test_reader_recovers_planted_text_in_both_formats(self):
        rng = np.random.default_rng(2)
        geom = {"xs": [6 + 20 * i for i in range(18)], "gw": 12, "y0": 5, "y1": 21}
        glyphs = [(rng.random((16, 12)) > 0.5).astype(np.float32) * 255 for _ in archive.GLYPHS]
        templates = np.stack([archive.norm(g) for g in glyphs])
        for text in ("2024-07-2713:42:45", "27-07-202413:42:45"):
            frame = np.zeros((32, 380), np.float32)
            for x, ch in zip(geom["xs"], text):
                frame[6:22, x + 1:x + 13] = glyphs[archive.GLYPHS.index(ch)]
            classes = archive.read_digits(frame[None].astype(np.uint8), geom, templates)[0]
            self.assertEqual("".join(archive.GLYPHS[c] for c in classes), text)
            offsets = archive.read_offsets(frame[None].astype(np.uint8), np.array([0.0]), geom, templates, START)
            self.assertEqual(offsets, [0.0])
            for sep in ((4, 7) if text[4] == "-" else (2, 5)) + (12, 15):
                x = geom["xs"][sep]
                frame[6:22, x + 1:x + 13] = glyphs[rng.integers(0, 10)]
            offsets = archive.read_offsets(frame[None].astype(np.uint8), np.array([0.0]), geom, templates, START)
            self.assertEqual(offsets, [0.0], f"separators must not matter for {text}")
        self.assertLessEqual(1, MARGIN)

    def test_known_layout_reused_without_tesseract(self):
        rng = np.random.default_rng(3)
        geom = {"xs": [6 + 20 * i for i in range(18)], "gw": 12, "y0": 5, "y1": 21}
        glyphs = [(rng.random((16, 12)) > 0.5).astype(np.float32) * 255 for _ in archive.GLYPHS]
        templates = np.stack([archive.norm(g) for g in glyphs])
        frames = np.zeros((30, 32, 380), np.uint8)
        pts = np.arange(0, 60, 2.0)
        for f, p in zip(frames, pts):
            text = (START + timedelta(seconds=p)).strftime("%d-%m-%Y%H:%M:%S")
            for x, ch in zip(geom["xs"], text):
                f[6:22, x + 1:x + 13] = glyphs[archive.GLYPHS.index(ch)]
        layouts = archive.Layouts(Path(os.environ["TAPO_DATA"]) / "none.npz")
        other = np.stack([archive.norm(rng.standard_normal((16, 12)).astype(np.float32)) for _ in archive.GLYPHS])
        layouts.models = {"v2_380x32_p20_h16": (templates, geom), "v2_380x32_p9_h16": (other, geom)}
        with mock.patch.object(archive, "detect_layout", side_effect=AssertionError("tesseract used")):
            chosen = layouts.get(frames, pts, 380, 32, START, "n")
        self.assertIs(chosen[0], templates)
        offsets = archive.read_offsets(frames, pts, geom, chosen[0], START)
        self.assertEqual(offsets, [0.0] * 30)

    def test_pick_shifts_finds_offset_and_matches_full_search(self):
        rng = np.random.default_rng(4)
        geom = {"xs": [6 + 20 * i for i in range(18)], "gw": 12, "y0": 5, "y1": 21}
        glyphs = [(rng.random((16, 12)) > 0.5).astype(np.float32) * 255 for _ in archive.GLYPHS]
        templates = np.stack([archive.norm(g) for g in glyphs])
        frames = np.zeros((40, 32, 380), np.uint8)
        for k, f in enumerate(frames):
            text = (START + timedelta(seconds=2 * k)).strftime("%Y-%m-%d%H:%M:%S")
            for x, ch in zip(geom["xs"], text):
                f[6:22, x + 1:x + 13] = glyphs[archive.GLYPHS.index(ch)]
        shifts = archive.pick_shifts(frames, geom, templates[:10])
        self.assertIn((MARGIN + 1, MARGIN + 1), shifts)
        self.assertLessEqual(len(shifts), 12)
        fast = archive.read_digits(frames, geom, templates[:10], shifts=shifts)
        full = archive.read_digits(frames, geom, templates[:10])
        digit_positions = archive.DATE_LAYOUTS[0][0] + archive.TIME_POS
        self.assertTrue((fast[:, digit_positions] == full[:, digit_positions]).all())
        offsets = archive.read_offsets(frames, np.arange(0, 80, 2.0), geom, templates, START)
        self.assertEqual(offsets, [0.0] * 40)

    def test_parse_overlay_formats(self):
        parse = archive.parse_overlay
        self.assertEqual(parse("2024-07-2713:42:45", START), START)
        self.assertEqual(parse("27-07-202413:42:45", START), START)
        self.assertEqual(parse("07-27-202413:42:45", START), START)
        self.assertEqual(parse("03-08-202401:00:00", START), datetime(2024, 8, 3, 1, 0, 0))
        self.assertIsNone(parse("2023-07-2713:42:45", START))
        self.assertIsNone(parse("2024-13-2713:42:45", START))
        self.assertTrue(archive.BOX_RE.fullmatch("22-06-202607:27:36"))
        self.assertTrue(archive.BOX_RE.fullmatch("2026-06-2117:28:41"))
        self.assertFalse(archive.BOX_RE.fullmatch("22-06-20260353310351"))


class FakeDownloadClient:
    def __init__(self, data):
        self.data = data
        self.calls = []

    async def iter_download(self, media, offset, limit, request_size, chunk_size, file_size):
        self.calls.append((offset, limit))
        self.assert_rules(offset, request_size)
        for i in range(limit):
            start = offset + i * request_size
            await asyncio.sleep(0)
            yield self.data[start:start + request_size]

    @staticmethod
    def assert_rules(offset, size):
        assert offset % 4096 == 0 and size % 4096 == 0 and (1024 * 1024) % size == 0


class FakeUploadClient:
    def __init__(self, fail_once=()):
        self.parts = {}
        self.fail_once = set(fail_once)
        self.small = None

    async def __call__(self, request):
        await asyncio.sleep(0)
        if request.file_part in self.fail_once:
            self.fail_once.discard(request.file_part)
            raise OSError("transient")
        self.parts[request.file_part] = (request.bytes, request.file_total_parts, request.file_id)
        return True

    async def upload_file(self, path):
        self.small = path
        return "small"


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="xfer-", dir=os.environ["TAPO_DATA"]))

    def test_parallel_download_reassembles(self):
        for size in (archive.PART_SIZE * 37 + 1234, archive.PART_SIZE * 3, 1000):
            data = os.urandom(size)
            client = FakeDownloadClient(data)
            path = self.dir / f"d{size}.bin"
            msg = mock.Mock(media=object())
            asyncio.run(archive.fast_download(client, msg, path, size))
            self.assertEqual(path.read_bytes(), data)
            self.assertLessEqual(len(client.calls), archive.DL_STREAMS)

    def test_parallel_upload_with_retry(self):
        data = os.urandom(archive.PART_SIZE * 25 + 77)
        path = self.dir / "u.bin"
        path.write_bytes(data)
        client = FakeUploadClient(fail_once={3, 17})
        with mock.patch.object(archive.asyncio, "sleep", new=mock.AsyncMock()):
            result = asyncio.run(archive.fast_upload(client, path))
        total = result.parts
        self.assertEqual(total, 26)
        self.assertEqual(b"".join(client.parts[i][0] for i in range(total)), data)
        self.assertEqual({p[1] for p in client.parts.values()}, {total})
        self.assertEqual(len({p[2] for p in client.parts.values()}), 1)

    def test_small_upload_uses_plain_path(self):
        path = self.dir / "s.bin"
        path.write_bytes(b"x" * 1000)
        client = FakeUploadClient()
        self.assertEqual(asyncio.run(archive.fast_upload(client, path)), "small")


class MediaAndCleanupTests(unittest.TestCase):
    def setUp(self):
        archive.WORK.mkdir(parents=True, exist_ok=True)
        self.src = archive.WORK / "1.mp4"
        make_video(self.src)

    def tearDown(self):
        archive.clear_work()

    def test_remux_faststart_aac_and_no_leftovers(self):
        out = archive.WORK / "1.faststart.mp4"
        events = [event(0, 6, START, START), event(6, 12, START + timedelta(hours=1), START)]
        archive.remux(self.src, out, events)
        duration, width, height, audio = archive.probe(out)
        self.assertAlmostEqual(duration, 12, delta=0.5)
        self.assertEqual(audio["codec_name"], "aac")
        data = out.read_bytes()
        self.assertIn(b"moov", data[:4096])
        self.assertLess(data.find(b"moov"), data.find(b"mdat"))
        chapters = subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-of", "csv=p=0", str(out)],
                                  capture_output=True, text=True).stdout
        self.assertEqual(len(chapters.strip().splitlines()), 2)
        self.assertEqual(sorted(p.name for p in archive.WORK.iterdir()), ["1.faststart.mp4", "1.mp4"])

    def test_cut_speech_lengths_and_cleanup(self):
        out = archive.WORK / "1.faststart.mp4"
        archive.remux(self.src, out, [event(0, 12, START, START)])
        clip = archive.WORK / "1.speech.mp4"
        lengths = archive.cut_speech(out, clip, [[0.0, 3.0], [6.0, 9.0]])
        self.assertEqual(len(lengths), 2)
        self.assertAlmostEqual(sum(lengths), archive.probe(clip)[0], delta=0.3)
        self.assertEqual(sorted(p.name for p in archive.WORK.iterdir()),
                         ["1.faststart.mp4", "1.mp4", "1.speech.mp4"])

    def test_cut_speech_from_alaw_source(self):
        clip = archive.WORK / "1.speech.mp4"
        lengths = archive.cut_speech(self.src, clip, [[0.0, 4.0], [8.0, 11.0]])
        self.assertAlmostEqual(sum(lengths), archive.probe(clip)[0], delta=0.3)
        self.assertEqual(archive.probe(clip)[3]["codec_name"], "aac")
        self.assertEqual(sorted(p.name for p in archive.WORK.iterdir()), ["1.mp4", "1.speech.mp4"])

    def test_cut_speech_failure_cleans_up(self):
        clip = archive.WORK / "1.speech.mp4"
        with self.assertRaises(RuntimeError):
            archive.cut_speech(archive.WORK / "missing.mp4", clip, [[0.0, 3.0]])
        self.assertEqual(sorted(p.name for p in archive.WORK.iterdir()), ["1.mp4"])

    def test_prepare_deletes_source_on_failure(self):
        item = {"id": 1, "name": "20240727_134245_tp00003.mp4", "start": START.isoformat()}
        with mock.patch.object(archive, "analyze", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                archive.prepare(item, self.src, None)
        self.assertEqual(list(archive.WORK.iterdir()), [])

    def test_prepare_success_leaves_only_outputs(self):
        item = {"id": 1, "name": "20240727_134245_tp00003.mp4", "start": START.isoformat()}
        segs = [event(0, 12, START, START + timedelta(seconds=12))]
        with mock.patch.object(archive, "analyze", return_value=(segs, segs, 1.0, [0.0, 2.0, 4.0])):
            r = archive.prepare(item, self.src, None)
        self.assertFalse(self.src.exists())
        self.assertEqual(sorted(p.name for p in archive.WORK.iterdir()), ["1.faststart.mp4", "1.opus"])
        self.assertEqual(archive.probe(r["audio"])[3]["codec_name"], "opus")
        self.assertEqual(r["keyframes"], [0.0, 2.0, 4.0])

    def test_clear_work_and_free(self):
        archive.clear_work()
        self.assertFalse(archive.WORK.exists())
        self.assertGreater(archive.free_gb(), 0)


class TranscriptionTests(unittest.TestCase):
    def test_plan_pieces_cut_at_quiet_frame_and_gate(self):
        fps = speech.FPS
        loud = np.where(np.arange(60 * fps) % 7 < 3, -20.0, -60.0)
        quiet = np.full(30 * fps, -60.0)
        levels = np.concatenate([loud, quiet])
        levels[18 * fps] = -90.0
        pieces = speech.plan_pieces(levels)
        self.assertEqual(pieces[0][1], 18 * fps + 1)
        self.assertEqual(pieces[0][0], 0)
        for a, b in zip(pieces, pieces[1:]):
            self.assertEqual(a[1], b[0])
        self.assertEqual(pieces[-1][1], len(levels))
        self.assertTrue(all(len(range(a, b)) <= speech.PIECE_S * fps for a, b, _ in pieces))
        self.assertTrue(pieces[0][2])
        self.assertFalse(pieces[-1][2])

    def test_split_phrases_words_and_noise(self):
        tokens = [" При", "вет", ",", " как", " де", "ла", "?", " Хо", "ро", "шо", "."]
        stamps = [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 3.0, 3.1, 3.2, 3.3]
        phrases = speech.split_phrases(tokens, stamps, 100.0)
        self.assertEqual([p[2] for p in phrases], ["Привет, как дела?", "Хорошо."])
        self.assertAlmostEqual(phrases[0][0], 100.0)
        self.assertAlmostEqual(phrases[1][0], 103.0)
        self.assertEqual(speech.split_phrases([" а", "."], [0.0, 0.1], 0.0), [])
        self.assertEqual(speech.split_phrases([" У", "гу", "."], [0.0, 0.1, 0.2], 0.0), [])
        self.assertEqual(speech.split_phrases([], [], 0.0), [])

    def test_transcribe_pipeline_and_decode_error(self):
        folder = Path(tempfile.mkdtemp(dir=os.environ["TAPO_DATA"]))
        tone = folder / "tone.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=8000:duration=50",
                        "-c:a", "pcm_alaw", str(tone)], check=True)
        rec = speech.Recognizer.__new__(speech.Recognizer)
        rec.lock = threading.Lock()
        seen = []

        def recognize(piece, sample_rate):
            seen.append(len(piece) / sample_rate)
            return mock.Mock(tokens=[" при", "вет", " мир"], timestamps=[0.0, 0.2, 0.5])

        rec.asr = mock.Mock(recognize=recognize)
        phrases = rec.transcribe(tone)
        self.assertAlmostEqual(sum(seen), 50.0, delta=0.2)
        self.assertTrue(all(s <= speech.PIECE_S + 0.01 for s in seen))
        self.assertEqual(phrases[0], (0.0, 0.8, "привет мир"))
        self.assertEqual(len(phrases), len(seen))
        with self.assertRaises(RuntimeError):
            rec.transcribe(folder / "missing.wav")

    def test_meaningful_filters_silence_hallucinations(self):
        keep = [["Спасибо."], ["ХХорошо."], ["КХонечно."], ["плати.", "Да."], ["Давай"], ["Чего", "заказ?"],
                ["В", "в", "чём", "проблема?"]]
        drop = [["Да."], ["Угу."], ["Да.", "Угу."], ["Агагу."], ["Я", "у."], ["ш", "вони."], ["аа."], ["Нет."], ["Дагу."]]
        for words in keep:
            self.assertTrue(speech.meaningful(words), words)
        for words in drop:
            self.assertFalse(speech.meaningful(words), words)

    def test_active_regions_and_compact_mapping(self):
        levels = np.full(200, -60.0)
        levels[50:53] = -20.0
        regions = speech.active_regions(levels, 200 * speech.FRAME)
        self.assertEqual(regions, [(45 * speech.FRAME, 58 * speech.FRAME)])
        piece = np.arange(200 * speech.FRAME, dtype=np.float32)
        audio, to_original = speech.compact(piece, regions + [(150 * speech.FRAME, 160 * speech.FRAME)])
        self.assertEqual(len(audio), 23 * speech.FRAME)
        self.assertAlmostEqual(to_original(0.0), 4.5)
        self.assertAlmostEqual(to_original(1.3), 15.0)
        self.assertAlmostEqual(to_original(1.4), 15.1)
        self.assertEqual(speech.active_regions(np.full(10, -30.0), 1000), [(0, 1000)])

    def test_quiet_piece_goes_through_vad_and_keeps_original_time(self):
        folder = Path(tempfile.mkdtemp(dir=os.environ["TAPO_DATA"]))
        quiet = folder / "quiet.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "sine=frequency=300:sample_rate=8000:duration=20,volume=0.0005", "-c:a", "pcm_alaw", str(quiet)],
                       check=True)
        rec = speech.Recognizer.__new__(speech.Recognizer)
        rec.lock = threading.Lock()
        rec.vad = mock.Mock()
        rec.vad.segment_batch.side_effect = lambda w, lens, sr, **kw: iter([iter([(5 * sr, 6 * sr)])])
        seen = []

        def recognize(audio, sample_rate):
            seen.append(len(audio) / sample_rate)
            return mock.Mock(tokens=[" в", " чём", " про", "бле", "ма", "?"], timestamps=[0.6, 0.8, 1.0, 1.1, 1.2, 1.3])

        rec.asr = mock.Mock(recognize=recognize)
        phrases = rec.transcribe(quiet)
        self.assertEqual(seen, [2.0])
        self.assertEqual(len(phrases), 1)
        self.assertAlmostEqual(phrases[0][0], 5.1, places=3)
        self.assertEqual(phrases[0][2], "в чём проблема?")

    def test_single_letter_phrase_dropped(self):
        tokens = [" Н", ".", " Толь", "ко", " со", "брал", "ся", "."]
        stamps = [0.0, 0.1, 2.5, 2.6, 2.8, 2.9, 3.0, 3.1]
        self.assertEqual([p[2] for p in speech.split_phrases(tokens, stamps, 0.0)], ["Только собрался."])

    def test_split_phrases_on_pause(self):
        tokens = [" да", "вай", " пой", "дём", " сей", "час"]
        stamps = [0.0, 0.2, 0.4, 0.6, 5.0, 5.2]
        phrases = speech.split_phrases(tokens, stamps, 0.0)
        self.assertEqual([p[2] for p in phrases], ["давай пойдём", "сейчас"])


class FakeClient:
    def __init__(self, role):
        self.role = role

    async def disconnect(self):
        pass


class ProcessLoopTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.captions = {}
        archive.clear_work()
        if archive.SPEECH_DIR.exists():
            shutil.rmtree(archive.SPEECH_DIR)

    def run_loop(self, failures_before=0, slow_first=False, phrases=None, broken_speech=(), initial=None, sounds=None):
        items = [
            {"id": i, "topic": 101, "name": f"20240727_00{i}000_tp0000{i}.mp4", "size": 10, "width": 1920,
             "height": 1080} for i in (1, 2, 3)
        ]
        archive.save_json(archive.SCAN_FILE, items)
        archive.save_json(archive.STATE_FILE, initial or ({"failures": {"1": failures_before}} if failures_before else {}))
        poster, loader = FakeClient("post"), FakeClient("download")
        calls = self.calls

        async def fake_connect(role):
            return (poster if role == "post" else loader), f"chat-{role}"

        async def fake_download(client, chat, item, path):
            assert client is loader and chat == "chat-download", "download must use the downloader bot"
            calls.append(("download", item["id"]))
            if item["id"] == 1 and failures_before:
                raise RuntimeError("broken source")
            path.write_bytes(b"x")
            return path

        def fake_prepare(item, src, layouts):
            if slow_first and item["id"] == 1:
                time.sleep(0.5)
            src.unlink()
            start = datetime.fromisoformat(item["start"])
            ev = [event(0, 10, start, start + timedelta(seconds=10))]
            out = archive.WORK / f"{item['id']}.faststart.mp4"
            out.write_bytes(b"v")
            audio = archive.WORK / f"{item['id']}.opus"
            audio.write_bytes(b"a")
            return {"start": start, "duration": 10.0, "width": 1920, "height": 1080, "out": out,
                    "segments": ev, "events": ev, "quality": 1.0, "keyframes": [0.0, 2.0, 4.0], "audio": audio}

        async def fake_ensure_topic(client, chat, state, name):
            assert client is poster and chat == "chat-post", "topics must use the posting bot"
            return {"Tapo C210": 201, "Tapo C210 субтитры": 203}[name]

        async def fake_upload(client, chat, topic, path, caption, rest, duration, width, height, name):
            assert client is poster and chat == "chat-post", "upload must use the posting bot"
            assert path.exists()
            calls.append(("upload", topic, name))
            self.captions[name] = caption + "\n" + "\n".join(rest)
            return [1000 + len(calls)]

        class FakeMessage:
            file = mock.Mock(size=5)

        async def fake_get_messages(chat, ids):
            assert chat == "chat-download"
            calls.append(("fetch posted", ids))
            return FakeMessage()

        async def fake_fast_download(client, msg, path, size):
            assert client is loader, "posted video must be fetched by the downloader bot"
            path.write_bytes(b"v")

        def fake_cut(src, dst, spans):
            dst.write_bytes(b"c")
            return [b - a for a, b in spans]

        recognizer = mock.Mock()
        def transcribe(path):
            if path.stem in broken_speech:
                raise RuntimeError("asr broke")
            return (phrases or {}).get(path.stem, [])

        def fake_rebuild(item, src, layouts, audio, meta_path, message):
            assert src.exists(), "rebuild must use the source downloaded by the downloader bot"
            calls.append(("rebuild", item["id"]))
            audio.write_bytes(b"a")
            start = datetime.fromisoformat(item["start"])
            ev = [event(0, 10, start, start + timedelta(seconds=10))]
            archive.save_json(meta_path, archive.speech_meta(
                {"start": start, "duration": 10.0, "width": 1920, "height": 1080, "segments": ev, "events": ev,
                 "keyframes": [0.0, 2.0]}, message))
            return True

        recognizer.transcribe.side_effect = transcribe
        detector = mock.Mock()
        detector.detect.side_effect = lambda path: (sounds or {}).get(path.stem, [])
        loader.get_messages = fake_get_messages
        args = mock.Mock(camera="", limit=0, dry_run=False)
        with mock.patch.object(archive, "connect", fake_connect), \
                mock.patch.object(archive, "download", fake_download), \
                mock.patch.object(archive, "prepare", fake_prepare), \
                mock.patch.object(archive, "Recognizer", mock.Mock(return_value=recognizer)), \
                mock.patch.object(archive, "SoundEvents", mock.Mock(return_value=detector)), \
                mock.patch.object(archive, "Layouts", mock.Mock()), \
                mock.patch.object(archive, "ensure_topic", fake_ensure_topic), \
                mock.patch.object(archive, "upload", fake_upload), \
                mock.patch.object(archive, "fast_download", fake_fast_download), \
                mock.patch.object(archive, "cut_speech", fake_cut), \
                mock.patch.object(archive, "rebuild_speech_input", fake_rebuild), \
                mock.patch.object(archive, "SPEECH_RETRY_DELAY", 0):
            asyncio.run(archive.cmd_process(args))
        return archive.load_json(archive.STATE_FILE, {})

    def test_roles_order_and_cleanup(self):
        state = self.run_loop(slow_first=True, phrases={"2": [(1.0, 2.0, "привет как дела")],
                                                        "3": [(3.0, 4.0, "пока пока")]})
        uploads = [c for c in self.calls if c[0] == "upload"]
        main = [c[2] for c in uploads if c[1] == 201]
        subs = [c[2] for c in uploads if c[1] == 203]
        self.assertEqual(sorted(c[1] for c in self.calls if c[0] == "download"), [1, 2, 3])
        self.assertEqual(main, ["20240727_001000_tp00001.mp4", "20240727_002000_tp00002.mp4",
                                "20240727_003000_tp00003.mp4"])
        self.assertEqual(subs, ["20240727_002000_tp00002_speech.mp4", "20240727_003000_tp00003_speech.mp4"])
        self.assertLess(self.calls.index(uploads[0]), self.calls.index(("upload", 201, main[1])))
        self.assertEqual(state["items"]["1"]["speech"], [])
        self.assertEqual(len(state["items"]["2"]["speech"]), 1)
        self.assertEqual(state["items"]["3"]["phrases"], 1)
        self.assertFalse(archive.WORK.exists())
        self.assertEqual(list(archive.SPEECH_DIR.iterdir()), [])

    def test_failure_below_limit_stops_run(self):
        with self.assertRaises(BaseException):
            self.run_loop(failures_before=1)
        state = archive.load_json(archive.STATE_FILE, {})
        self.assertEqual(state["failures"]["1"], 2)
        self.assertEqual(state.get("items", {}), {})
        self.assertFalse(archive.WORK.exists())

    def test_third_failure_skips_and_continues_in_order(self):
        state = self.run_loop(failures_before=2)
        self.assertIn("broken source", state["items"]["1"]["skipped"])
        main = [c[2] for c in self.calls if c[0] == "upload" and c[1] == 201]
        self.assertEqual(main, ["20240727_002000_tp00002.mp4", "20240727_003000_tp00003.mp4"])
        self.assertFalse(archive.WORK.exists())


class SoundAndMaskTests(ProcessLoopTests):
    def test_scream_only_file_gets_subtitle_post_and_words_masked(self):
        self.run_loop(phrases={"3": [(1.0, 2.0, "началась война в Украине")]},
                      sounds={"2": [(4.0, 6.0, "крик", 0.8)], "3": [(0.0, 1.5, "стон", 0.5)]})
        subs = {c[2] for c in self.calls if c[0] == "upload" and c[1] == 203}
        self.assertEqual(subs, {"20240727_002000_tp00002_speech.mp4", "20240727_003000_tp00003_speech.mp4"})
        self.assertIn("[крик]", self.captions["20240727_002000_tp00002_speech.mp4"])
        text = self.captions["20240727_003000_tp00003_speech.mp4"]
        self.assertIn("началась в***а в У*****е", text)
        self.assertNotIn("войн", text)
        self.assertLess(text.index("[стон]"), text.index("началась"))


class BlacklistTests(unittest.TestCase):
    def test_masking(self):
        from blacklist import mask
        cases = {
            "Россия и Украина, война.": "Р****я и У*****а, в***а.",
            "РФ": "Р*", "Путина?": "П****а?", "премьер-министра": "п*****р-м******а",
            "бомбёжка": "б******а", "no war": "no w*r", "ВСУ и НАТО": "В*У и Н**О",
            "ракетка армированный воин своё Warsaw танкер": "ракетка армированный воин своё Warsaw танкер",
            "плёнка плен": "плёнка п**н",
            "Путин и Зеленского, Байдену, Лукашенко": "П***н и З********о, Б*****у, Л*******о",
            "Горбачёв Сталина Trump": "Г******в С*****а T***p",
            "кулебяка бандероль кличка бушлат маска Ермаковский": "кулебяка бандероль кличка бушлат маска Ермаковский",
            "Кулеба Бандера Кличко Буша": "К****а Б*****а К****о Б**а",
        }
        for src, want in cases.items():
            self.assertEqual(mask(src), want, src)

    def test_tos_categories(self):
        from blacklist import mask
        cases = {
            "Блять, охуеть, наебал, распиздяй, сука!": "Б***ь, о****ь, н****л, р*******й, с**а!",
            "Потрахаться, трахал, секс, порно": "П*********я, т****л, с**с, п***о",
            "героин, мефедрон, закладчик": "г****н, м******н, з*******к",
            "Хач, жиды, свастика, ИГИЛ": "Х*ч, ж**ы, с******а, И**Л",
            "педофил, изнасилование, самоубийство": "п*****л, и***********е, с**********о",
            "fuck shit porn": "f**k s**t p**n",
            "Звони +7 (912) 345-67-89": "Звони +7 (***) ***-**-*9",
            "карта 4276 1234 5678 9012": "карта 4*** **** **** ***2",
        }
        for src, want in cases.items():
            self.assertEqual(mask(src), want, src)

    def test_tos_no_false_hits(self):
        from blacklist import mask
        text = ("себе небо хлеба вебинар вебка страха страхует мандарин хачапури колеблясь рубля требовать ребята "
                "героиня анализ канал сукно художник хулиган method 80% в 2024 году, в 12 30")
        self.assertEqual(mask(text), text)


class SoundEventTests(unittest.TestCase):
    def test_mel_matrix(self):
        import sounds
        m = sounds.mel_matrix()
        self.assertEqual(m.shape, (257, 64))
        self.assertTrue((m[0] == 0).all())
        self.assertTrue((m.sum(axis=0) > 0).all())

    def test_find_events_grouping(self):
        import sounds
        groups = ["крик", "стон", "плач"]
        times = [i * 0.48 for i in range(12)]
        rows = np.zeros((12, 3))
        rows[2:5, 0] = 0.5
        rows[7, 1] = 0.7
        rows[9, 2] = 0.4
        events = sounds.find_events(times, rows, groups)
        levels_mel, levels = sounds.log_mel(np.zeros(16000, np.float32), sounds.mel_matrix(),
                                            np.hanning(sounds.WINDOW).astype(np.float32))
        self.assertEqual(len(levels_mel), len(levels))
        self.assertTrue((levels < sounds.QUIET_DB).all())
        self.assertEqual([(round(a, 2), round(b, 2), g) for a, b, g, _ in events],
                         [(0.96, 2.88, "крик"), (3.36, 4.32, "стон")])


class SpeechResilienceTests(ProcessLoopTests):
    def test_speech_failure_does_not_block_main(self):
        state = self.run_loop(phrases={"3": [(1.0, 2.0, "привет как дела")]}, broken_speech={"2"})
        main = [c[2] for c in self.calls if c[0] == "upload" and c[1] == 201]
        subs = [c[2] for c in self.calls if c[0] == "upload" and c[1] == 203]
        self.assertEqual(len(main), 3)
        self.assertEqual(subs, ["20240727_003000_tp00003_speech.mp4"])
        self.assertEqual(state["speech_failures"]["2"], archive.SPEECH_MAX_FAILURES)
        self.assertIn("asr broke", state["items"]["2"]["speech_skipped"])
        self.assertNotIn("speech_skipped", state["items"]["3"])
        self.assertFalse(archive.WORK.exists())

    def test_missing_speech_input_rebuilt_from_source(self):
        initial = {"items": {"1": {"camera": "Tapo C210", "name": "20240727_001000_tp00001.mp4", "messages": [77]}}}
        state = self.run_loop(phrases={"1": [(1.0, 2.0, "это старый файл")]}, initial=initial)
        self.assertIn(("rebuild", 1), self.calls)
        self.assertLess(self.calls.index(("download", 1)), self.calls.index(("rebuild", 1)))
        main = [c[2] for c in self.calls if c[0] == "upload" and c[1] == 201]
        self.assertEqual(len(main), 2)
        self.assertEqual(len(state["items"]["1"]["speech"]), 1)


class LimitAndRecoveryTests(unittest.TestCase):
    def test_widest_known_font_fits_crop_and_overflow_is_rejected(self):
        cw, ch = archive.crop_size(2304, 1296)
        geom = {"xs": [6 + 44 * c for c in archive.CELLS], "gw": 28, "y0": 4, "y1": 50}
        frames = np.zeros((2, ch, cw), np.uint8)
        templates = np.ones((10, geom["y1"] - geom["y0"], geom["gw"]), np.float32)
        self.assertEqual(archive.read_digits(frames, geom, templates).shape, (2, len(archive.CELLS)))
        start = datetime(2024, 1, 1)
        text = "2024-01-0100:00:00"

        def reads(pitch, y1=30):
            boxes = [(4 + pitch * c, 4, 4 + pitch * c + 20, y1) for c in archive.CELLS]
            return mock.patch.object(archive, "tesseract_boxes", return_value=(start, text, boxes))

        small = np.zeros((5, 60, 400), np.uint8)
        with reads(10):
            self.assertIsNotNone(archive.detect_layout(small, 800, 1000, start))
        with reads(30):
            self.assertIsNone(archive.detect_layout(small, 800, 1000, start))
        with reads(10, y1=70):
            self.assertIsNone(archive.detect_layout(small, 800, 1000, start))

    def test_frame_subset_reads_only_requested_frames(self):
        frames = np.arange(40, dtype=np.uint8).reshape(10, 2, 2)
        subset = archive.FrameSubset(frames, np.array([1, 3, 5, 7, 9]))
        self.assertEqual(len(subset), 5)
        self.assertEqual(subset.shape, (5, 2, 2))
        self.assertTrue((subset[1] == frames[3]).all())
        self.assertTrue((subset[1:3] == frames[[3, 5]]).all())
        self.assertTrue((subset[::2] == frames[[1, 5, 9]]).all())
        self.assertEqual(len(list(subset)), 5)

    def test_memory_gate_waits_then_gives_up(self):
        tmp = Path(tempfile.mkdtemp(prefix="tapo-mem-"))
        try:
            (tmp / "max").write_text("1000\n")
            (tmp / "stat").write_text("file 5\nanon 900\nkernel 1\n")
            with mock.patch.object(archive, "MEMORY_MAX_FILE", tmp / "max"), \
                    mock.patch.object(archive, "MEMORY_STAT_FILE", tmp / "stat"), \
                    mock.patch.object(archive, "MEMORY_WAIT", 0):
                self.assertEqual(archive.memory_share(), 0.9)
                asyncio.run(archive.wait_for_memory("x"))
            (tmp / "max").write_text("max\n")
            with mock.patch.object(archive, "MEMORY_MAX_FILE", tmp / "max"), \
                    mock.patch.object(archive, "MEMORY_STAT_FILE", tmp / "stat"):
                self.assertEqual(archive.memory_share(), 0.0)
            with mock.patch.object(archive, "MEMORY_MAX_FILE", tmp / "absent"):
                self.assertEqual(archive.memory_share(), 0.0)
        finally:
            shutil.rmtree(tmp)

    def test_analyze_reads_mixed_layouts_in_one_file(self):
        n = 200
        frames = np.zeros((n, 4, 4), np.uint8)
        frames[60:, 0, 0] = 1
        pts = np.arange(n, dtype=float) * 2
        big, small = (np.zeros(1), {"name": "big"}), (np.zeros(1), {"name": "small"})
        asked = []

        class FakeLayouts:
            def get(self, sub, sub_pts, width, height, start, name):
                asked.append(len(sub))
                return big if len(sub) == n else small

        def fake_offsets(sub, sub_pts, geom, templates, start):
            want = 0 if geom["name"] == "big" else 1
            return [0.0 if f[0, 0] == want else None for f in sub]

        with mock.patch.object(archive, "overlay_frames", return_value=(frames, pts)), \
                mock.patch.object(archive, "read_offsets", fake_offsets):
            segments, events, quality, _ = archive.analyze(Path("x.mp4"), 4, 4, START, n * 2, FakeLayouts(), "x")
        self.assertEqual(asked, [n, n - 60])
        self.assertGreater(quality, 0.95)
        with mock.patch.object(archive, "overlay_frames", return_value=(frames, pts)), \
                mock.patch.object(archive, "read_offsets", fake_offsets):
            asked.clear()
            single = mock.Mock()
            single.get.return_value = big
            _, _, quality, _ = archive.analyze(Path("x.mp4"), 4, 4, START, n * 2, single, "x")
        self.assertEqual(single.get.call_count, 2)
        self.assertLess(quality, 0.5)

    def test_tesseract_fallback_scale_and_trailing_junk(self):
        start = datetime(2024, 10, 3, 1, 29, 32)
        calls = []

        def fake_read(img, k):
            calls.append((k, int(img.max()), int(img.min())))
            if len(calls) == 1:
                return "2024-1", [(0, 0, 1, 1)] * 6
            text = "2024-10-0301:43:507-"
            return text, [(i, 0, i + 1, 1) for i in range(len(text))]

        frame = np.full((42, 400), 150, np.uint8)
        frame[10:20, 10:20] = 250
        with mock.patch.object(archive, "tesseract_read", fake_read):
            dt, chars, boxes = archive.tesseract_boxes(frame, start)
        self.assertEqual(dt, datetime(2024, 10, 3, 1, 43, 50))
        self.assertEqual(chars, "2024-10-0301:43:50")
        self.assertEqual(len(boxes), 18)
        self.assertEqual([c[0] for c in calls], [3, 3])
        self.assertEqual(calls[1][1:], (255, 0))
        calls.clear()
        with mock.patch.object(archive, "tesseract_read", lambda img, k: calls.append(k) or ("", [])):
            self.assertIsNone(archive.tesseract_boxes(np.zeros((76, 820), np.uint8), start))
        self.assertEqual(calls, [2, 2, 3, 3])

    def test_overlay_frames_reads_keyframes_and_cleans_up(self):
        archive.WORK.mkdir(parents=True, exist_ok=True)
        src = archive.WORK / "kf.mp4"
        make_video(src, seconds=12, audio=False)
        try:
            frames, pts = archive.overlay_frames(src, 320, 240)
            cw, ch = archive.crop_size(320, 240)
            self.assertEqual(frames.shape, (6, ch, cw))
            self.assertEqual(len(pts), 6)
            self.assertEqual(list(archive.WORK.glob("*.raw")), [])
        finally:
            src.unlink()

    def make_state(self):
        return {"items": {
            "1": {"camera": "Tapo C210", "name": "20240901_000000_tp1.mp4", "messages": [1], "speech": [11, 12]},
            "2": {"camera": "Tapo C210", "name": "20240902_000000_tp2.mp4", "messages": [2], "speech": [],
                  "speech_skipped": "MediaCaptionTooLongError"},
            "3": {"camera": "Tapo C200", "name": "20240903_000000_tp3.mp4", "messages": [3], "speech": [31]},
            "4": {"camera": "Tapo C210", "name": "20240904_000000_tp4.mp4", "messages": [4], "speech": [41, 42],
                  "phrases": 3},
            "5": {"camera": "Tapo C210", "name": "20240905_000000_tp5.mp4", "messages": [5]},
        }, "speech_failures": {"2": 5, "3": 1}, "failures": {"4": 1}}

    def run_reset(self, state, speech_only):
        deleted = []

        class Client:
            async def delete_messages(self, chat, ids):
                deleted.extend(ids)

            async def disconnect(self):
                pass

        async def fake_connect(role):
            assert role == "post"
            return Client(), "chat"

        with mock.patch.object(archive, "connect", fake_connect):
            asyncio.run(archive.reset_from(state, "20240902_000000_tp2.mp4", speech_only))
        return sorted(deleted), archive.load_json(archive.STATE_FILE, {})

    def test_reset_from_main_redoes_posts_and_drops_queue_files(self):
        archive.SPEECH_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("4.mp4", "4.json", "1.mp4"):
            (archive.SPEECH_DIR / name).write_bytes(b"x")
        deleted, saved = self.run_reset(self.make_state(), False)
        self.assertEqual(deleted, [2, 4, 5, 41, 42])
        self.assertEqual(sorted(saved["items"]), ["1", "3"])
        self.assertEqual(saved["speech_failures"], {"3": 1})
        self.assertEqual(saved["failures"], {})
        self.assertEqual(sorted(p.name for p in archive.SPEECH_DIR.iterdir()), ["1.mp4"])
        shutil.rmtree(archive.SPEECH_DIR)
        archive.STATE_FILE.unlink()

    def test_reset_from_speech_only_redoes_later_files_of_same_camera(self):
        deleted, saved = self.run_reset(self.make_state(), True)
        self.assertEqual(deleted, [41, 42])
        self.assertNotIn("speech", saved["items"]["2"])
        self.assertNotIn("speech_skipped", saved["items"]["2"])
        self.assertNotIn("speech", saved["items"]["4"])
        self.assertEqual(saved["items"]["1"]["speech"], [11, 12])
        self.assertEqual(saved["items"]["3"]["speech"], [31])
        self.assertEqual(saved["speech_failures"], {"3": 1})
        archive.STATE_FILE.unlink()


if __name__ == "__main__":
    unittest.main()
