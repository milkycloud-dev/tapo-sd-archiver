import argparse
import asyncio
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from telethon import TelegramClient, functions
from telethon.errors import RPCError
from telethon.tl.functions.upload import SaveBigFilePartRequest
from telethon.tl.types import DocumentAttributeFilename, DocumentAttributeVideo, InputFileBig

from backup import CAPTION_LIMIT, DATA, FFMPEG, MESSAGE_LIMIT, ROOT, fmt_duration, fmt_offset, load_env, log, run_ffmpeg
from blacklist import mask
from sounds import SoundEvents
from speech import Recognizer

def env_pairs(name):
    return [tuple(x.strip() for x in part.split("=", 1)) for part in os.environ.get(name, "").split(",") if "=" in part]


load_env(ROOT / ".env")
CHAT = int(os.environ.get("TG_CHAT", "0") or 0)
TOPIC_CAMERA = {int(topic): camera for topic, camera in env_pairs("ARCHIVE_SOURCES")}
SOURCE_TOPICS = tuple(TOPIC_CAMERA)
TOPICS = {name: int(topic) for name, topic in env_pairs("ARCHIVE_TOPICS")}
SCAN_FILE = DATA / "scan.json"
STATE_FILE = DATA / "archive_state.json"
TEMPLATES_FILE = DATA / "layouts.npz"
WORK = DATA / "archive_work"
ROLES = {"post": ("TG_BOT_TOKEN", "bot"), "download": ("TG_DL_BOT_TOKEN", "bot_dl")}
SPEECH_DIR = DATA / "speech_queue"
RUNNING = set()
SUBTITLE_SUFFIX = " субтитры"
SPEECH_PAD = 2.0
SPEECH_JOIN = 8.0
BOX_RE = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})\d{2}:\d{2}:\d{2}")
GLYPHS = "0123456789-:"
TESS_HEIGHT = 120
TEXT_LEVEL = 200
CALIBRATE_FRAMES = 300
MEMORY_GATE = float(os.environ.get("MEMORY_GATE", "0.6") or 0.6)
MEMORY_WAIT = 600
MEMORY_MAX_FILE = Path("/sys/fs/cgroup/memory.max")
MEMORY_STAT_FILE = Path("/sys/fs/cgroup/memory.stat")
MAX_LAYOUTS = 3
MIN_UNREAD = 20
OVERLAY_FORMATS = ("%Y-%m-%d%H:%M:%S", "%d-%m-%Y%H:%M:%S", "%m-%d-%Y%H:%M:%S")
CELLS = list(range(10)) + list(range(11, 19))
TIME_POS = [10, 11, 13, 14, 16, 17]
DATE_LAYOUTS = (
    ([0, 1, 2, 3, 5, 6, 8, 9], ("%Y%m%d%H%M%S",)),
    ([0, 1, 3, 4, 6, 7, 8, 9], ("%d%m%Y%H%M%S", "%m%d%Y%H%M%S")),
)
NAME_RE = re.compile(r"(\d{8}_\d{6})_tp\d+\.mp4")
MARGIN = 3
ALL_SHIFTS = [(dy, dx) for dy in range(2 * MARGIN + 1) for dx in range(2 * MARGIN + 1)]
MERGE_GAP = 120
PREFETCH = int(os.environ.get("ARCHIVE_PREFETCH", "3") or 3)
ASR_THREADS = int(os.environ.get("ASR_THREADS", "1") or 1)
WORKERS = int(os.environ.get("ARCHIVE_WORKERS", "2") or 2)
DL_STREAMS = int(os.environ.get("DL_STREAMS", "8") or 8)
UL_STREAMS = int(os.environ.get("UL_STREAMS", "8") or 8)
PART_SIZE = 512 * 1024
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "6") or 6)
KNOWN_LAYOUT_SCORE = 0.5
MAX_FAILURES = 3
SPEECH_QUEUE_GB = float(os.environ.get("SPEECH_QUEUE_GB", "6") or 6)
SPEECH_MAX_FAILURES = 5
SPEECH_RETRY_DELAY = 60


async def connect(role):
    if not CHAT or not TOPIC_CAMERA:
        sys.exit("TG_CHAT and ARCHIVE_SOURCES must be set in .env")
    token, session = ROLES[role]
    client = TelegramClient(
        str(DATA / session), int(os.environ["TG_API_ID"]), os.environ["TG_API_HASH"],
        flood_sleep_threshold=600,
    )
    await client.start(bot_token=os.environ[token])
    return client, await client.get_entity(CHAT)


def topic_of(msg):
    rt = msg.reply_to
    if not rt or not getattr(rt, "forum_topic", False):
        return 0
    return rt.reply_to_top_id or rt.reply_to_msg_id or 0


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


async def cmd_scan(args):
    client, chat = await connect("download")
    found, empty, msg_id = [], 0, args.start
    try:
        while empty < args.stop_after:
            batch = await client.get_messages(chat, ids=list(range(msg_id, msg_id + 100)))
            for msg in batch:
                if not msg:
                    empty += 1
                    continue
                empty = 0
                topic = topic_of(msg)
                if topic not in SOURCE_TOPICS or not msg.file:
                    continue
                f = msg.file
                found.append({
                    "id": msg.id, "topic": topic, "posted": msg.date.isoformat(),
                    "name": f.name, "size": f.size, "mime": f.mime_type,
                    "duration": getattr(f, "duration", None), "width": f.width, "height": f.height,
                    "camera": TOPIC_CAMERA[topic],
                    "caption": msg.message or "", "grouped_id": msg.grouped_id,
                })
            msg_id += 100
            if msg_id % 1000 == args.start % 1000:
                log(f"scan at {msg_id}, {len(found)} files")
    finally:
        await client.disconnect()
    save_json(SCAN_FILE, found)
    log(f"scan done: last id {msg_id - empty}, {len(found)} files -> {SCAN_FILE}")
    summary(found)


def summary(found):
    print(Counter((x["topic"], x["camera"]) for x in found))
    print(Counter(x["mime"] for x in found))
    for cam in sorted({x["camera"] for x in found}):
        items = [x for x in found if x["camera"] == cam]
        hours = sum(x["duration"] or 0 for x in items) / 3600
        size = sum(x["size"] or 0 for x in items) / 1e9
        names = sorted(x["name"] or "" for x in items)
        print(f"{cam}: {len(items)} files, {hours:.1f} h, {size:.1f} GB, {names[0]} .. {names[-1]}")


def build_queue(found):
    videos = sorted((x for x in found if NAME_RE.fullmatch(x["name"] or "")), key=lambda x: x["id"])
    seen, queue, skipped = set(), [], []
    for x in videos:
        x["camera"] = TOPIC_CAMERA[x["topic"]]
        key = (x["camera"], x["name"])
        if key in seen:
            skipped.append(x)
            continue
        seen.add(key)
        x["start"] = datetime.strptime(NAME_RE.fullmatch(x["name"])[1], "%Y%m%d_%H%M%S").isoformat()
        queue.append(x)
    queue.sort(key=lambda x: (x["start"], x["id"]))
    return queue, skipped


def norm(g):
    g = g - g.mean()
    n = np.linalg.norm(g)
    return g / n if n else g


def crop_size(width, height):
    return min(width, 960) // 2 * 2, int(height * 0.06) // 2 * 2


class FrameSubset:
    def __init__(self, frames, index):
        self.frames, self.index = frames, index
        self.shape = (len(index), *frames.shape[1:])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        return np.asarray(self.frames[self.index[key]])


def overlay_frames(path, width, height):
    cw, ch = crop_size(width, height)
    raw = Path(path).with_suffix(".overlay.raw")
    try:
        proc = subprocess.run(
            [FFMPEG, "-hide_banner", "-v", "info", "-y", "-skip_frame", "nokey", "-i", str(path), "-an",
             "-vf", f"crop={cw}:{ch}:0:0,format=gray,showinfo", "-fps_mode", "passthrough",
             "-f", "rawvideo", str(raw)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        frames = np.zeros(0, np.uint8)
        if raw.exists() and raw.stat().st_size:
            frames = np.memmap(raw, np.uint8, mode="r")
    finally:
        try:
            raw.unlink(missing_ok=True)
        except OSError:
            pass
    frames = frames[:len(frames) // (ch * cw) * ch * cw].reshape(-1, ch, cw)
    pts = [float(x) for x in re.findall(rb"pts_time:([\d.]+)", proc.stderr)]
    if not len(frames) or len(pts) != len(frames):
        raise RuntimeError(f"overlay extraction failed: {len(frames)} frames, {len(pts)} pts")
    return frames, np.array(pts)


def tesseract_read(img, k):
    big = np.repeat(np.repeat(img, k, 0), k, 1)
    h, w = big.shape
    with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as f:
        f.write(f"P5 {w} {h} 255\n".encode() + big.tobytes())
    try:
        proc = subprocess.run(
            ["tesseract", f.name, "-", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789-:", "makebox"],
            capture_output=True, text=True, env={**os.environ, "OMP_THREAD_LIMIT": "1"},
        )
    finally:
        os.unlink(f.name)
    chars, boxes = "", []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and len(parts[0]) == 1 and parts[0] in GLYPHS:
            left, bottom, right, top = (int(v) for v in parts[1:5])
            chars += parts[0]
            boxes.append((left / k, (h - top) / k, right / k, (h - bottom) / k))
    return chars, boxes


def tesseract_boxes(frame, start):
    base = max(2, round(TESS_HEIGHT / frame.shape[0]))
    for k in (base, base + 1):
        for img in (255 - frame, np.where(frame > TEXT_LEVEL, 0, 255).astype(np.uint8)):
            chars, boxes = tesseract_read(img, k)
            found = BOX_RE.match(chars)
            if found:
                dt = parse_overlay(found.group(), start)
                if dt:
                    return dt, found.group(), boxes[:found.end()]
    return None


def parse_overlay(text, start):
    best = None
    for fmt in OVERLAY_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        delta = (dt - start).total_seconds()
        if -86400 <= delta <= 10 * 86400 and (best is None or abs(delta) < abs((best - start).total_seconds())):
            best = dt
    return best


def detect_layout(frames, width, height, start, picks=40):
    reads = [r for i in np.linspace(0, len(frames) - 1, min(picks, len(frames))).astype(int)
             if (r := tesseract_boxes(frames[i], start))]
    fits = []
    for _, chars, boxes in reads:
        centers = np.array([(b[0] + b[2]) / 2 for b in boxes])
        pitch, x0 = np.polyfit(CELLS, centers, 1)
        if np.abs(centers - (x0 + pitch * np.array(CELLS))).max() < 0.25 * pitch:
            fits.append((pitch, x0, chars, boxes))
    if len(fits) < 3:
        return None
    pitch = float(np.median([f[0] for f in fits]))
    x0 = float(np.median([f[1] for f in fits]))
    widths = [b[2] - b[0] for _, _, chars, boxes in fits for c, b in zip(chars, boxes) if c.isdigit() and c != "1"]
    tops = [b[1] for _, _, chars, boxes in fits for c, b in zip(chars, boxes) if c.isdigit()]
    bottoms = [b[3] for _, _, chars, boxes in fits for c, b in zip(chars, boxes) if c.isdigit()]
    gw = int(np.median(widths)) + 2
    y0, y1 = max(int(np.median(tops)) - 1, 0), int(np.ceil(np.median(bottoms))) + 1
    xs = [int(round(x0 + pitch * c - gw / 2)) for c in CELLS]
    if min(xs) < 0 or xs[-1] + gw > frames.shape[2] or y1 > frames.shape[1]:
        return None
    signature =f"v2_{width}x{height}_p{pitch:.0f}_h{(y1 - y0) // 2 * 2}"
    return signature, {"xs": xs, "gw": gw, "y0": y0, "y1": y1}


def glyph_windows(frame, geom):
    gh, gw = geom["y1"] - geom["y0"], geom["gw"]
    padded = np.pad(frame, MARGIN * 2)
    y = geom["y0"] + MARGIN
    return [padded[y:y + gh + 2 * MARGIN, x + MARGIN:x + MARGIN + gw + 2 * MARGIN] for x in geom["xs"]]


def glyph_shifts(win, geom):
    gh, gw = geom["y1"] - geom["y0"], geom["gw"]
    return np.stack([
        norm(win[dy:dy + gh, dx:dx + gw])
        for dy in range(2 * MARGIN + 1) for dx in range(2 * MARGIN + 1)
    ])


def calibrate(frames, pts, start, geom):
    gh, gw = geom["y1"] - geom["y0"], geom["gw"]
    reads = []
    for i in range(0, len(frames), max(3, len(frames) // CALIBRATE_FRAMES)):
        r = tesseract_boxes(frames[i], start)
        if r:
            reads.append((i, r[1], (r[0] - start).total_seconds() - pts[i]))
    labeled = []
    for k, (i, chars, off) in enumerate(reads):
        if any(abs(o - off) <= 1.5 for j, (_, _, o) in enumerate(reads) if j != k and abs(j - k) <= 3):
            labeled.append((frames[i].astype(np.float32), [GLYPHS.index(c) for c in chars]))
    acc = [[] for _ in GLYPHS]
    for frame, lab in labeled:
        for win, d in zip(glyph_windows(frame, geom), lab):
            acc[d].append(norm(win[MARGIN:MARGIN + gh, MARGIN:MARGIN + gw]))
    missing = [GLYPHS[d] for d, a in enumerate(acc) if not a]
    if missing:
        raise RuntimeError(f"calibration: no samples of {missing} in {len(labeled)} labeled frames")
    base = np.stack([norm(np.mean(a, 0)) for a in acc])
    for _ in range(2):
        acc = [[] for _ in GLYPHS]
        for frame, lab in labeled:
            for win, d in zip(glyph_windows(frame, geom), lab):
                shifted = glyph_shifts(win, geom)
                acc[d].append(shifted[np.einsum("sij,ij->s", shifted, base[d]).argmax()])
        base = np.stack([norm(np.mean(a, 0)) for a in acc])
    return base, len(labeled)


class Layouts:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.models = {}
        if path.exists():
            with np.load(path) as data:
                for key in data.files:
                    if key.endswith("|t"):
                        sig = key[:-2]
                        g = data[sig + "|g"].tolist()
                        self.models[sig] = (data[key], {"y0": g[0], "y1": g[1], "gw": g[2], "xs": g[3:]})

    def score(self, frames, pts, start, model):
        step = max(len(frames) // 150, 1)
        offsets = read_offsets(frames[::step], pts[::step], model[1], model[0], start)
        good = sum(1 for k, o in enumerate(offsets) if o is not None and any(
            offsets[j] is not None and abs(offsets[j] - o) <= 2 for j in (k - 1, k + 1) if 0 <= j < len(offsets)))
        return good / max(len(offsets), 1)

    def get(self, frames, pts, width, height, start, name):
        prefix = f"v2_{width}x{height}_"
        with self.lock:
            known = [m for sig, m in self.models.items() if sig.startswith(prefix)]
        scored = sorted(((self.score(frames, pts, start, m), i) for i, m in enumerate(known)), reverse=True)
        if scored and scored[0][0] >= KNOWN_LAYOUT_SCORE:
            return known[scored[0][1]]
        found = detect_layout(frames, width, height, start) or detect_layout(frames, width, height, start, 160)
        if not found:
            with self.lock:
                known = [m for sig, m in self.models.items() if sig.startswith(prefix)]
            scored = sorted(((self.score(frames, pts, start, m), i) for i, m in enumerate(known)), reverse=True)
            return known[scored[0][1]] if scored and scored[0][0] > 0 else (None, None)
        signature, geom = found
        with self.lock:
            if signature not in self.models:
                templates, count = calibrate(frames, pts, start, geom)
                self.models[signature] = (templates, geom)
                data = {}
                for sig, (t, g) in self.models.items():
                    data[sig + "|t"] = t
                    data[sig + "|g"] = np.array([g["y0"], g["y1"], g["gw"], *g["xs"]])
                tmp = self.path.with_name(self.path.stem + ".tmp.npz")
                np.savez(tmp, **data)
                tmp.replace(self.path)
                log(f"  new overlay layout {signature} calibrated on {name} ({count} frames)")
            return self.models[signature]


def pad_frames(frames):
    return np.pad(frames.astype(np.float32), ((0, 0), (2 * MARGIN, 2 * MARGIN), (2 * MARGIN, 2 * MARGIN)))


def shift_scores(padded, geom, templates, dy, dx):
    gh, gw = geom["y1"] - geom["y0"], geom["gw"]
    y = geom["y0"] + MARGIN
    g = np.stack([padded[:, y + dy:y + dy + gh, x + MARGIN + dx:x + MARGIN + dx + gw] for x in geom["xs"]], axis=1)
    g -= g.mean(axis=(2, 3), keepdims=True)
    n = np.linalg.norm(g, axis=(2, 3), keepdims=True)
    g /= np.where(n == 0, 1, n)
    return np.einsum("fpij,dij->fpd", g, templates)


def read_digits(frames, geom, templates, batch=64, shifts=ALL_SHIFTS):
    out = []
    for b in range(0, len(frames), batch):
        padded = pad_frames(frames[b:b + batch])
        best = np.full((len(padded), len(geom["xs"]), len(templates)), -np.inf, np.float32)
        for dy, dx in shifts:
            np.maximum(best, shift_scores(padded, geom, templates, dy, dx), out=best)
        out.append(best.argmax(-1))
    return np.concatenate(out)


def pick_shifts(frames, geom, templates, sample=64):
    padded = pad_frames(frames[::max(len(frames) // sample, 1)][:sample])
    peaks = np.stack([shift_scores(padded, geom, templates, dy, dx).max(-1) for dy, dx in ALL_SHIFTS])
    confident = peaks.max(0) >= 0.5
    if not confident.any():
        return ALL_SHIFTS
    winners = Counter(peaks.argmax(0)[confident].tolist())
    total = sum(winners.values())
    chosen = [ALL_SHIFTS[k] for k, n in winners.items() if n >= 0.02 * total]
    return sorted({(dy + a, dx + b) for dy, dx in chosen for a in (-1, 0, 1) for b in (-1, 0, 1)
                   if 0 <= dy + a <= 2 * MARGIN and 0 <= dx + b <= 2 * MARGIN})


def parse_digits(row, start):
    clock = "".join(str(d) for d in row[TIME_POS])
    best = None
    for positions, formats in DATE_LAYOUTS:
        date = "".join(str(d) for d in row[positions])
        for fmt in formats:
            try:
                dt = datetime.strptime(date + clock, fmt)
            except ValueError:
                continue
            delta = (dt - start).total_seconds()
            if -86400 <= delta <= 10 * 86400 and (best is None or abs(delta) < abs((best - start).total_seconds())):
                best = dt
    return best


def read_offsets(frames, pts, geom, templates, start):
    offsets = []
    digits = templates[:10]
    for row, p in zip(read_digits(frames, geom, digits, shifts=pick_shifts(frames, geom, digits)), pts):
        dt = parse_digits(row, start)
        offsets.append((dt - start).total_seconds() - p if dt else None)
    return offsets


def build_segments(offsets, pts, start, duration):
    valid = [k for k, o in enumerate(offsets) if o is not None]
    segments, current = [], None
    for n, k in enumerate(valid):
        o = offsets[k]
        if current is not None and abs(o - current) <= 2:
            segments[-1]["last"] = k
            continue
        ahead = valid[n + 1:n + 3]
        if len(ahead) == 2 and all(abs(offsets[j] - o) <= 2 for j in ahead):
            current = o
            segments.append({"first": k, "last": k, "offset": o})
    if not segments or abs(segments[0]["offset"]) > 3:
        segments.insert(0, {"first": 0, "last": 0, "offset": 0.0})
    segments[0]["first"] = 0
    out = []
    for i, seg in enumerate(segments):
        pos = float(pts[seg["first"]])
        end = float(pts[segments[i + 1]["first"]]) if i + 1 < len(segments) else duration
        cam = start + timedelta(seconds=seg["offset"] + pos)
        out.append({"pos": pos, "end": end, "cam": cam, "cam_end": cam + timedelta(seconds=end - pos)})
    consistent = sum(1 for k, o in enumerate(offsets) if o is not None and any(
        offsets[j] is not None and abs(offsets[j] - o) <= 2 for j in (k - 1, k + 1) if 0 <= j < len(offsets)))
    return out, consistent / len(offsets)


def merge_events(segments):
    events = []
    for seg in segments:
        if events and 0 <= (seg["cam"] - events[-1]["cam_end"]).total_seconds() < MERGE_GAP:
            events[-1]["end"] = seg["end"]
            events[-1]["cam_end"] = seg["cam_end"]
            continue
        events.append(dict(seg))
    return events


def plural(n, one, few, many):
    if n % 10 == 1 and n % 100 != 11:
        return one
    return few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many


def split_text(header, lines):
    caption, i = header, 0
    while i < len(lines) and len(caption) + 1 + len(lines[i]) <= CAPTION_LIMIT:
        caption += "\n" + lines[i]
        i += 1
    rest, chunk = [], ""
    for line in lines[i:]:
        while len(line) > MESSAGE_LIMIT:
            if chunk:
                rest.append(chunk)
                chunk = ""
            rest.append(line[:MESSAGE_LIMIT])
            line = line[MESSAGE_LIMIT:]
        if chunk and len(chunk) + 1 + len(line) > MESSAGE_LIMIT:
            rest.append(chunk)
            chunk = ""
        chunk += ("\n" if chunk else "") + line
    if chunk:
        rest.append(chunk)
    return caption, rest


def time_span(start, duration, events):
    first = events[0]["cam"] if events else start
    last = events[-1]["cam_end"] if events else start + timedelta(seconds=duration)
    if first.date() == last.date():
        return first, f"{first:%d.%m.%Y %H:%M} - {last:%H:%M}"
    return first, f"{first:%d.%m.%Y %H:%M} - {last:%d.%m.%Y %H:%M}"


def cam_label(moment, first):
    return f"{moment:%H:%M:%S}" if moment.date() == first.date() else f"{moment:%d.%m %H:%M:%S}"


def build_caption(start, duration, events, note=""):
    first, span = time_span(start, duration, events)
    n = len(events)
    header = f"{span}\n{fmt_duration(duration)} видео {n} {plural(n, 'событие', 'события', 'событий')}{note}"
    header += "\nвремя в ролике - время камеры:"
    lines = [f"{fmt_offset(e['pos'])} - {cam_label(e['cam'], first)}" for e in events]
    return split_text(header, lines)


def cam_at(segments, pos):
    seg = segments[0]
    for s in segments:
        if s["pos"] <= pos:
            seg = s
    return seg["cam"] + timedelta(seconds=pos - seg["pos"])


def speech_spans(phrases, duration, keyframes):
    spans = []
    for a, b, _ in phrases:
        a, b = max(a - SPEECH_PAD, 0.0), min(b + SPEECH_PAD, duration)
        a = max((k for k in keyframes if k <= a + 0.01), default=0.0)
        if spans and a - spans[-1][1] < SPEECH_JOIN:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return spans


def cut_speech(src, dst, spans):
    parts = []
    listing = dst.with_suffix(".txt")
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    try:
        for i, (a, b) in enumerate(spans):
            part = dst.with_name(f"{dst.stem}.{i:03}.mp4")
            parts.append(part)
            run_ffmpeg("-ss", f"{a:.3f}", "-i", src, "-t", f"{b - a:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
                       "-c:v", "copy", "-c:a", "aac", "-b:a", "32k", "-ar", 16000, "-ac", 1,
                       "-avoid_negative_ts", "make_zero", part)
        lengths = [probe(p)[0] for p in parts]
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        run_ffmpeg("-f", "concat", "-safe", 0, "-i", listing, "-c", "copy", "-movflags", "+faststart", tmp)
        tmp.replace(dst)
        return lengths
    finally:
        for path in (*parts, listing, tmp):
            path.unlink(missing_ok=True)


def build_speech_caption(start, duration, events, segments, phrases, spans, lengths, clip_len, link):
    first, span = time_span(start, duration, events)
    header = (f"{span}\nречь {fmt_duration(clip_len)} из {fmt_duration(duration)} видео\n"
              f"полное видео: {link}\nвремя в ролике - время камеры - текст:")
    offsets, total = [], 0.0
    for length in lengths:
        offsets.append(total)
        total += length
    lines = []
    for a, _, text in phrases:
        i = max(j for j, (s0, _) in enumerate(spans) if s0 <= a + 0.01)
        pos = offsets[i] + a - spans[i][0]
        lines.append(f"{fmt_offset(pos)} - {cam_label(cam_at(segments, a), first)} {text}")
    return split_text(header, lines)


def probe(path):
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,width,height",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    info = json.loads(proc.stdout or "{}")
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    return float(info.get("format", {}).get("duration", 0)), video.get("width", 0), video.get("height", 0), audio


def remux(src, dst, events):
    _, _, _, audio = probe(src)
    meta = [";FFMETADATA1"]
    for i, e in enumerate(events):
        end = events[i + 1]["pos"] if i + 1 < len(events) else e["end"]
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(e['pos'] * 1000)}", f"END={int(end * 1000)}",
                 f"title={e['cam']:%d.%m %H:%M:%S}"]
    meta_file = dst.with_suffix(".ffmeta")
    audio_args = ["-map", "0:a:0", "-c:a", "aac", "-b:a", "32k", "-ar", 16000, "-ac", 1] if audio else []
    tmp = dst.with_name(dst.stem + ".tmp.mp4")
    try:
        meta_file.write_text("\n".join(meta) + "\n", encoding="utf-8")
        run_ffmpeg(
            "-i", src, "-i", meta_file, "-map", "0:v:0", *audio_args, "-c:v", "copy",
            "-map_metadata", -1, "-map_chapters", 1, "-movflags", "+faststart", tmp,
        )
        tmp.replace(dst)
    finally:
        meta_file.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)


def analyze(path, width, height, start, duration, layouts, name):
    frames, pts = overlay_frames(path, width, height)
    offsets, todo, used = [None] * len(frames), np.arange(len(frames)), []
    for _ in range(MAX_LAYOUTS):
        sub, sub_pts = (frames, pts) if len(todo) == len(frames) else (FrameSubset(frames, todo), pts[todo])
        try:
            templates, geom = layouts.get(sub, sub_pts, width, height, start, name)
        except RuntimeError as exc:
            log(f"  {name}: {exc}")
            break
        if geom is None or any(g is geom for g in used):
            break
        used.append(geom)
        for i, o in zip(todo, read_offsets(sub, sub_pts, geom, templates, start)):
            offsets[i] = o
        todo = np.array([i for i in todo if offsets[i] is None], dtype=int)
        if len(todo) < max(MIN_UNREAD, 0.1 * len(frames)):
            break
    segments, quality = build_segments(offsets, pts, start, duration)
    return segments, merge_events(segments), quality, [float(p) for p in pts]


async def fast_download(client, msg, path, size):
    chunks = math.ceil(size / PART_SIZE)
    per = math.ceil(chunks / DL_STREAMS)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)

    async def worker(first, count):
        pos = first * PART_SIZE
        async for data in client.iter_download(
            msg.media, offset=pos, limit=count, request_size=PART_SIZE, chunk_size=PART_SIZE, file_size=size,
        ):
            os.pwrite(fd, data, pos)
            pos += len(data)
        return pos - first * PART_SIZE

    try:
        os.ftruncate(fd, size)
        got = await asyncio.gather(*(
            worker(i * per, min(per, chunks - i * per)) for i in range(DL_STREAMS) if i * per < chunks
        ))
    finally:
        os.close(fd)
    if sum(got) != size:
        raise RuntimeError(f"got {sum(got)} of {size} bytes")


async def download(client, chat, item, path):
    if path.exists():
        return path
    part = path.with_suffix(".part")
    for attempt in range(3):
        try:
            msg = await client.get_messages(chat, ids=item["id"])
            if not msg or not msg.file:
                raise RuntimeError("source message has no file")
            t0 = time.monotonic()
            await fast_download(client, msg, part, item["size"])
            log(f"  downloaded {item['name']} at {item['size'] / 1e6 / (time.monotonic() - t0):.1f} MB/s")
            if part.stat().st_size != item["size"]:
                raise RuntimeError(f"size {part.stat().st_size} != {item['size']}")
            part.replace(path)
            return path
        except Exception as exc:
            log(f"  download {item['id']} {item['name']}: {type(exc).__name__}: {exc}")
            await asyncio.sleep(30 * (attempt + 1))
    part.unlink(missing_ok=True)
    raise RuntimeError(f"download failed: {item['name']}")


async def ensure_topic(client, chat, state, camera):
    topics = state.setdefault("topics", {})
    if camera in TOPICS:
        topics[camera] = TOPICS[camera]
    if camera in topics:
        return topics[camera]
    request = getattr(functions.messages, "CreateForumTopicRequest", None) or functions.channels.CreateForumTopicRequest
    params = {"title": camera}
    if request.__module__.endswith("messages"):
        params["peer"] = chat
    else:
        params["channel"] = chat
    result = await client(request(**params))
    topic = next(u.message.id for u in result.updates if getattr(getattr(u, "message", None), "id", None))
    topics[camera] = topic
    save_json(STATE_FILE, state)
    log(f"created topic '{camera}' id {topic}")
    return topic


async def fast_upload(client, path):
    size = path.stat().st_size
    if size < 10 * 1024 * 1024:
        return await client.upload_file(str(path))
    total = math.ceil(size / PART_SIZE)
    file_id = random.getrandbits(63)
    parts = iter(range(total))
    fd = os.open(path, os.O_RDONLY)

    async def worker():
        for index in parts:
            data = os.pread(fd, PART_SIZE, index * PART_SIZE)
            for attempt in range(5):
                try:
                    if await client(SaveBigFilePartRequest(file_id, index, total, data)):
                        break
                except (OSError, RPCError) as exc:
                    log(f"  part {index}/{total}: {type(exc).__name__}: {exc}")
                await asyncio.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"upload part {index} failed")

    try:
        await asyncio.gather(*(worker() for _ in range(UL_STREAMS)))
    finally:
        os.close(fd)
    return InputFileBig(file_id, total, path.name)


async def upload(client, chat, topic, path, caption, rest, duration, width, height, name):
    thumb = path.with_suffix(".jpg")
    try:
        await in_thread(run_ffmpeg, "-ss", 1, "-i", path, "-frames:v", 1, "-vf", "scale=320:-2", thumb)
        t0 = time.monotonic()
        uploaded = await fast_upload(client, path)
        log(f"  uploaded {name} at {path.stat().st_size / 1e6 / (time.monotonic() - t0):.1f} MB/s")
        attributes = [
            DocumentAttributeVideo(duration=duration, w=width, h=height, supports_streaming=True),
            DocumentAttributeFilename(name),
        ]
        msg = await client.send_file(
            chat, uploaded, caption=caption, parse_mode=None, reply_to=topic, thumb=str(thumb),
            supports_streaming=True, force_document=False, mime_type="video/mp4", attributes=attributes,
        )
    finally:
        thumb.unlink(missing_ok=True)
    ids = [msg.id]
    for text in rest:
        ids.append((await client.send_message(chat, text, reply_to=msg.id, parse_mode=None)).id)
    return ids


def free_gb():
    return shutil.disk_usage(DATA).free / 1e9


def clear_work():
    shutil.rmtree(WORK, ignore_errors=True)


def settle(fut):
    RUNNING.discard(fut)
    if not fut.cancelled():
        fut.exception()


async def in_thread(fn, *args):
    fut = asyncio.get_running_loop().run_in_executor(None, fn, *args)
    RUNNING.add(fut)
    fut.add_done_callback(settle)
    return await asyncio.shield(fut)


async def drain_threads(timeout=50):
    if RUNNING:
        await asyncio.wait(set(RUNNING), timeout=timeout)


def memory_share():
    try:
        limit = int(MEMORY_MAX_FILE.read_text())
        stat = MEMORY_STAT_FILE.read_text()
        return int(re.search(r"^anon (\d+)", stat, re.M)[1]) / limit
    except (OSError, ValueError, TypeError):
        return 0.0


async def wait_for_memory(name):
    deadline = time.monotonic() + MEMORY_WAIT
    warned = False
    while memory_share() >= MEMORY_GATE and time.monotonic() < deadline:
        if not warned:
            log(f"  {name}: waiting for memory, {memory_share():.0%} of the limit in use")
            warned = True
        await asyncio.sleep(5)


def dir_gb(path):
    return sum(p.stat().st_size for p in path.glob("*") if p.is_file()) / 1e9 if path.exists() else 0.0


def dump_timeline(items):
    return [{"pos": e["pos"], "end": e["end"], "cam": e["cam"].isoformat(), "cam_end": e["cam_end"].isoformat()}
            for e in items]


def load_timeline(items):
    return [{"pos": e["pos"], "end": e["end"], "cam": datetime.fromisoformat(e["cam"]),
             "cam_end": datetime.fromisoformat(e["cam_end"])} for e in items]


def extract_audio(src, dst):
    run_ffmpeg("-i", src, "-vn", "-ac", 1, "-ar", 16000, "-c:a", "libopus", "-b:a", "24k", dst)


def prepare(item, src, layouts):
    out = WORK / f"{item['id']}.faststart.mp4"
    audio = WORK / f"{item['id']}.opus"
    try:
        start = datetime.fromisoformat(item["start"])
        duration, width, height, has_audio = probe(src)
        t0 = time.monotonic()
        segments, events, quality, keyframes = analyze(src, width, height, start, duration, layouts, item["name"])
        if has_audio:
            extract_audio(src, audio)
        remux(src, out, events)
        src.unlink()
        log(f"  prepared {item['name']}: {len(events)} events, quality {quality:.3f}, {time.monotonic() - t0:.0f}s")
        return {
            "start": start, "duration": duration, "width": width, "height": height, "out": out,
            "segments": segments, "events": events, "quality": quality, "keyframes": keyframes,
            "audio": audio if has_audio else None,
        }
    except BaseException:
        for path in (src, out, audio):
            path.unlink(missing_ok=True)
        raise


def speech_files(run, key):
    return run.speech_dir / f"{key}.opus", run.speech_dir / f"{key}.json", run.speech_dir / f"{key}.mp4"


def message_link(message_id):
    return f"https://t.me/c/{str(CHAT)[4:]}/{message_id}"


class Run:
    def __init__(self, args, queue):
        self.args = args
        self.state = load_json(STATE_FILE, {})
        self.done = self.state.setdefault("items", {})
        self.items = queue
        if args.camera:
            self.items = [x for x in self.items if args.camera.lower() in x["camera"].lower()]
        self.posted_all = asyncio.Event()
        self.speech_dir = WORK / "dry_speech" if args.dry_run else SPEECH_DIR

    def save(self):
        if not self.args.dry_run:
            save_json(STATE_FILE, self.state)

    def fail(self, item, exc, key, label):
        failures = self.state.setdefault(key, {})
        failures[str(item["id"])] = failures.get(str(item["id"]), 0) + 1
        log(f"{label} {item['name']} failed ({failures[str(item['id'])]}): {type(exc).__name__}: {exc}")
        self.save()
        return failures[str(item["id"])] >= MAX_FAILURES and not self.args.dry_run


async def main_stream(run, poster, post_chat, loader, load_chat, layouts):
    pending = [x for x in run.items if not run.done.get(str(x["id"]), {}).get("messages")
               and "skipped" not in run.done.get(str(x["id"]), {})]
    if run.args.limit:
        pending = pending[:run.args.limit]
    log(f"main stream: {len(pending)} to post")
    net = asyncio.Semaphore(PREFETCH)
    cpu = asyncio.Semaphore(WORKERS)
    tasks, scheduled = {}, set()

    async def fetch_and_prepare(item):
        async with net:
            warned = False
            while free_gb() < MIN_FREE_GB or dir_gb(run.speech_dir) > SPEECH_QUEUE_GB:
                if not warned:
                    log(f"  waiting: free {free_gb():.1f} GB, speech queue {dir_gb(run.speech_dir):.1f} GB")
                    warned = True
                await asyncio.sleep(30)
            src = await download(loader, load_chat, item, WORK / f"{item['id']}.mp4")
        async with cpu:
            await wait_for_memory(item["name"])
            return await in_thread(prepare, item, src, layouts)

    def schedule(upto):
        for item in pending[:upto]:
            if item["id"] not in scheduled:
                scheduled.add(item["id"])
                tasks[item["id"]] = asyncio.create_task(fetch_and_prepare(item))

    try:
        for n, item in enumerate(pending):
            schedule(n + WORKERS + PREFETCH)
            key = str(item["id"])
            try:
                r = await tasks.pop(item["id"])
            except Exception as exc:
                if not run.fail(item, exc, "failures", f"[{n + 1}/{len(pending)}]"):
                    raise
                run.done[key] = {"camera": item["camera"], "name": item["name"], "speech": [],
                                 "skipped": f"{type(exc).__name__}: {exc}"[:300]}
                run.save()
                continue
            caption, rest = build_caption(r["start"], r["duration"], r["events"])
            log(f"[{n + 1}/{len(pending)}] {item['camera']} {item['name']}")
            try:
                if run.args.dry_run:
                    print(caption)
                    for text in rest:
                        print("--- reply ---\n" + text)
                    ids = [0]
                else:
                    topic = await ensure_topic(poster, post_chat, run.state, item["camera"])
                    ids = await upload(poster, post_chat, topic, r["out"], caption, rest, r["duration"],
                                       r["width"], r["height"], item["name"])
                entry = run.done[key] = {
                    "camera": item["camera"], "name": item["name"], "messages": ids,
                    "events": len(r["events"]), "quality": round(r["quality"], 3),
                    "posted": datetime.now().isoformat(timespec="seconds"),
                }
                if r["audio"]:
                    run.speech_dir.mkdir(parents=True, exist_ok=True)
                    meta = speech_meta(r, ids[0])
                    audio_path, meta_path, video_path = speech_files(run, key)
                    r["out"].replace(video_path)
                    r["audio"].replace(audio_path)
                    save_json(meta_path, meta)
                else:
                    entry["speech"] = []
                    entry["phrases"] = 0
                run.save()
            finally:
                r["out"].unlink(missing_ok=True)
                if r["audio"]:
                    r["audio"].unlink(missing_ok=True)
    finally:
        run.posted_all.set()
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)


def speech_meta(r, message):
    return {
        "start": r["start"].isoformat(), "duration": r["duration"], "width": r["width"], "height": r["height"],
        "segments": dump_timeline(r["segments"]), "events": dump_timeline(r["events"]),
        "keyframes": r["keyframes"], "message": message,
    }


def rebuild_speech_input(item, src, layouts, audio, meta_path, message):
    start = datetime.fromisoformat(item["start"])
    duration, width, height, has_audio = probe(src)
    segments, events, quality, keyframes = analyze(src, width, height, start, duration, layouts, item["name"])
    if has_audio:
        extract_audio(src, audio)
    r = {"start": start, "duration": duration, "width": width, "height": height,
         "segments": segments, "events": events, "keyframes": keyframes}
    save_json(meta_path, speech_meta(r, message))
    return has_audio


async def speech_item(run, item, entry, poster, post_chat, loader, load_chat, recognizer, detector, layouts, work):
    key = str(item["id"])
    audio, meta_path, video = speech_files(run, key)
    src, clip = work / f"{key}.mp4", work / f"{key}.speech.mp4"
    try:
        if not meta_path.exists() or not audio.exists():
            log(f"  speech {item['name']}: rebuilding input from source")
            await download(loader, load_chat, item, src)
            run.speech_dir.mkdir(parents=True, exist_ok=True)
            if not await in_thread(rebuild_speech_input, item, src, layouts, audio, meta_path, entry["messages"][0]):
                entry["speech"] = []
                entry["phrases"] = 0
                run.save()
                meta_path.unlink(missing_ok=True)
                return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        await wait_for_memory(item["name"])
        t0 = time.monotonic()
        spoken = [(a, b, mask(text)) for a, b, text in await in_thread(recognizer.transcribe, audio)]
        sounds = await in_thread(detector.detect, audio)
        phrases = sorted(spoken + [(a, b, f"[{label}]") for a, b, label, _ in sounds])
        log(f"  speech {item['name']}: {len(spoken)} phrases, {sum(len(p[2].split()) for p in spoken)} words, "
            f"{len(sounds)} sound events, {time.monotonic() - t0:.0f}s")
        speech_ids = []
        if phrases:
            start, duration = datetime.fromisoformat(meta["start"]), meta["duration"]
            segments, events = load_timeline(meta["segments"]), load_timeline(meta["events"])
            spans = speech_spans(phrases, duration, meta["keyframes"])
            source = video if video.exists() else await download(loader, load_chat, item, src)
            lengths = await in_thread(cut_speech, source, clip, spans)
            clip_len = sum(lengths)
            caption, rest = build_speech_caption(start, duration, events, segments, phrases, spans, lengths,
                                                 clip_len, message_link(meta["message"]))
            if run.args.dry_run:
                print("=== speech ===\n" + caption)
                for text in rest:
                    print("--- reply ---\n" + text)
            else:
                topic = await ensure_topic(poster, post_chat, run.state, item["camera"] + SUBTITLE_SUFFIX)
                speech_ids = await upload(poster, post_chat, topic, clip, caption, rest, clip_len,
                                          meta["width"], meta["height"], item["name"].replace(".mp4", "_speech.mp4"))
        entry["speech"] = speech_ids
        entry["phrases"] = len(spoken)
        entry["sounds"] = len(sounds)
        run.save()
        for path in (audio, meta_path, video):
            path.unlink(missing_ok=True)
    finally:
        src.unlink(missing_ok=True)
        clip.unlink(missing_ok=True)


async def speech_stream(run, poster, post_chat, loader, load_chat, recognizer, detector, layouts):
    work = WORK / "speech"
    work.mkdir(parents=True, exist_ok=True)
    for item in run.items:
        key = str(item["id"])
        while True:
            entry = run.done.get(key, {})
            if "speech" in entry or entry.get("messages") or "skipped" in entry:
                break
            if run.posted_all.is_set():
                return
            await asyncio.sleep(2)
        if "speech" in entry:
            continue
        while True:
            try:
                await speech_item(run, item, entry, poster, post_chat, loader, load_chat, recognizer, detector, layouts,
                                  work)
                break
            except Exception as exc:
                failures = run.state.setdefault("speech_failures", {})
                failures[key] = failures.get(key, 0) + 1
                log(f"speech {item['name']} failed ({failures[key]}/{SPEECH_MAX_FAILURES}): {type(exc).__name__}: {exc}")
                if failures[key] >= SPEECH_MAX_FAILURES or run.args.dry_run:
                    entry["speech"] = []
                    entry["speech_skipped"] = f"{type(exc).__name__}: {exc}"[:300]
                    run.save()
                    for path in speech_files(run, key):
                        path.unlink(missing_ok=True)
                    break
                run.save()
                await asyncio.sleep(SPEECH_RETRY_DELAY * failures[key])


async def cmd_process(args):
    os.nice(10)
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    found = load_json(SCAN_FILE, None)
    if found is None:
        sys.exit("run scan first")
    queue, skipped = build_queue(found)
    run = Run(args, queue)
    log(f"queue {len(queue)} (skipped {len(skipped)}), workers {WORKERS}, asr threads {ASR_THREADS}, "
        f"streams down {DL_STREAMS} up {UL_STREAMS}")
    clear_work()
    WORK.mkdir(parents=True, exist_ok=True)
    SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    for stray in SPEECH_DIR.glob("*.tmp"):
        stray.unlink()
    poster, post_chat = await connect("post")
    loader, load_chat = await connect("download")
    try:
        layouts = Layouts(TEMPLATES_FILE)
        recognizer = await asyncio.to_thread(Recognizer, ASR_THREADS)
        detector = await asyncio.to_thread(SoundEvents, 1)
        async with asyncio.TaskGroup() as group:
            group.create_task(main_stream(run, poster, post_chat, loader, load_chat, layouts))
            group.create_task(speech_stream(run, poster, post_chat, loader, load_chat, recognizer, detector, layouts))
    finally:
        await drain_threads()
        await poster.disconnect()
        await loader.disconnect()
        clear_work()
        log(f"work dir cleared, free {free_gb():.1f} GB, speech queue {dir_gb(SPEECH_DIR):.2f} GB")


async def reset_from(state, name, speech_only):
    items = state.get("items", {})
    target = next((e for e in items.values() if e["name"] == name), None)
    if target is None:
        sys.exit(f"{name} is not in the archive state")
    chosen = {k: e for k, e in items.items() if e["camera"] == target["camera"] and e["name"] >= name
              and ("speech" in e or not speech_only)}
    ids = [m for e in chosen.values() for m in (e.get("speech") or []) + ([] if speech_only else e.get("messages") or [])]
    poster, chat = await connect("post")
    try:
        for i in range(0, len(ids), 100):
            await poster.delete_messages(chat, ids[i:i + 100])
    finally:
        await poster.disconnect()
    for key, entry in chosen.items():
        state.get("speech_failures", {}).pop(key, None)
        if speech_only:
            for field in ("speech", "phrases", "sounds", "speech_skipped"):
                entry.pop(field, None)
            continue
        state.get("failures", {}).pop(key, None)
        items.pop(key)
        for path in SPEECH_DIR.glob(f"{key}.*"):
            path.unlink()
    save_json(STATE_FILE, state)
    log(f"reset: {len(ids)} {'speech ' if speech_only else ''}posts of {len(chosen)} {target['camera']} files "
        f"from {name} deleted, they will be redone in order")


async def cmd_reset(args):
    if not args.yes:
        sys.exit("reset deletes every post of the posting bot in the archive topics; pass --yes")
    state = load_json(STATE_FILE, {})
    if args.start:
        await reset_from(state, args.start, args.speech_only)
        return
    names = {**TOPICS, **state.get("topics", {})}
    topics = {t for name, t in names.items() if not args.speech_only or name.endswith(SUBTITLE_SUFFIX)}
    poster, chat = await connect("post")
    me = await poster.get_me()
    try:
        first = min(topics)
        ids, empty, msg_id = [], 0, first
        while empty < 300:
            batch = await poster.get_messages(chat, ids=list(range(msg_id, msg_id + 100)))
            for msg in batch:
                if not msg:
                    empty += 1
                    continue
                empty = 0
                if msg.sender_id == me.id and msg.id not in topics and topic_of(msg) in topics:
                    ids.append(msg.id)
            msg_id += 100
        log(f"reset: {len(ids)} posts of {me.username} in topics {sorted(topics)}")
        for i in range(0, len(ids), 100):
            await poster.delete_messages(chat, ids[i:i + 100])
    finally:
        await poster.disconnect()
    if args.speech_only:
        for entry in state.get("items", {}).values():
            for key in ("speech", "phrases", "sounds", "speech_skipped"):
                entry.pop(key, None)
        state.pop("speech_failures", None)
        save_json(STATE_FILE, state)
        log("reset: speech results cleared, main posts kept")
        return
    state["items"] = {}
    for key in ("failures", "speech_failures"):
        state.pop(key, None)
    save_json(STATE_FILE, state)
    if SPEECH_DIR.exists():
        shutil.rmtree(SPEECH_DIR)
    log("reset: state cleared")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--start", type=int, default=min(SOURCE_TOPICS, default=0))
    scan.add_argument("--stop-after", type=int, default=1000)
    sub.add_parser("summary")
    process = sub.add_parser("process")
    process.add_argument("--limit", type=int, default=0)
    process.add_argument("--camera", default="")
    process.add_argument("--dry-run", action="store_true")
    reset = sub.add_parser("reset")
    reset.add_argument("--yes", action="store_true")
    reset.add_argument("--speech-only", action="store_true")
    reset.add_argument("--from", dest="start", default="")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env(DATA / ".env")
    if args.cmd == "summary":
        found = load_json(SCAN_FILE, [])
        summary(found)
        queue, skipped = build_queue(found)
        print(Counter(x["camera"] for x in queue), "skipped", [(x["id"], x["name"], x["camera"]) for x in skipped])
        return
    try:
        asyncio.run({"scan": cmd_scan, "process": cmd_process, "reset": cmd_reset}[args.cmd](args))
    except asyncio.CancelledError:
        log("stopped by signal")
        sys.exit(143)


if __name__ == "__main__":
    main()
