import hashlib
import io
import os
import threading
import urllib.request
import zipfile

import numpy as np

from backup import DATA, log
from speech import READ_S, SR, audio_stream, finish

YAMNET_URL = ("https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/yamnet/releases/"
              "v0.65.0/yamnet-onnx-float.zip")
YAMNET_SHA256 = "cf08212c3b9c03160d3c343a5cf99dd53edcb3176b49b6dd330d7d0480abd309"
YAMNET_DIR = DATA / "models" / "yamnet-v0.65.0"
WINDOW, HOP, FFT, MELS = 400, 160, 512, 64
PATCH, PATCH_HOP = 96, 48
EVENTS = {
    "крик": ("Screaming", "Yell"),
    "стон": ("Groan", "Wail, moan", "Whimper"),
    "плач": ("Crying, sobbing", "Baby cry, infant cry"),
}
THRESHOLD = float(os.environ.get("EVENT_THRESHOLD", "0.35") or 0.35)
STRONG = float(os.environ.get("EVENT_STRONG", "0.6") or 0.6)
MERGE_GAP = 2.0
QUIET_DB = float(os.environ.get("EVENT_QUIET_DB", "-48") or -48)


def fetch_yamnet():
    model = YAMNET_DIR / "yamnet.onnx"
    if model.exists() and (YAMNET_DIR / "yamnet.data").exists() and (YAMNET_DIR / "labels.txt").exists():
        return YAMNET_DIR
    with urllib.request.urlopen(YAMNET_URL, timeout=120) as resp:
        data = resp.read()
    got = hashlib.sha256(data).hexdigest()
    if got != YAMNET_SHA256:
        raise RuntimeError(f"yamnet: sha256 {got} != {YAMNET_SHA256}")
    YAMNET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in ("yamnet.onnx", "yamnet.data", "labels.txt"):
            tmp = YAMNET_DIR / (name + ".part")
            tmp.write_bytes(z.read(f"yamnet-onnx-float/{name}"))
            tmp.replace(YAMNET_DIR / name)
    log("model yamnet v0.65.0 verified")
    return YAMNET_DIR


def hz_to_mel(f):
    return 1127.0 * np.log1p(np.asarray(f, dtype=np.float64) / 700.0)


def mel_matrix():
    bins = FFT // 2 + 1
    spec_mel = hz_to_mel(np.linspace(0, SR / 2, bins)[1:])[:, None]
    edges = np.linspace(hz_to_mel(125.0), hz_to_mel(7500.0), MELS + 2)
    lower, center, upper = edges[:-2], edges[1:-1], edges[2:]
    weights = np.maximum(0.0, np.minimum((spec_mel - lower) / (center - lower), (upper - spec_mel) / (upper - center)))
    return np.vstack([np.zeros((1, MELS)), weights]).astype(np.float32)


def log_mel(samples, matrix, window):
    count = (len(samples) - WINDOW) // HOP + 1
    if count <= 0:
        return np.zeros((0, MELS), np.float32), np.zeros(0, np.float32)
    idx = np.arange(WINDOW)[None, :] + HOP * np.arange(count)[:, None]
    framed = samples[idx]
    spectrum = np.abs(np.fft.rfft(framed * window, FFT)).astype(np.float32)
    levels = 20 * np.log10(np.sqrt((framed.astype(np.float64) ** 2).mean(1)) + 1e-9)
    return np.log(spectrum @ matrix + 0.001), levels.astype(np.float32)


def find_events(times, scores, labels):
    marks = []
    for t, row in zip(times, scores):
        best = int(np.argmax(row))
        marks.append((t, labels[best], float(row[best])))
    events, run = [], []
    for t, label, score in marks + [(None, None, 0.0)]:
        if score >= THRESHOLD and (not run or run[-1][1] == label and t - run[-1][0] <= PATCH_HOP / 100 + 0.01):
            run.append((t, label, score))
            continue
        if run and (len(run) >= 2 or max(s for _, _, s in run) >= STRONG):
            start, end = run[0][0], run[-1][0] + PATCH / 100
            if events and events[-1][2] == run[0][1] and start - events[-1][1] < MERGE_GAP:
                events[-1] = (events[-1][0], end, events[-1][2], max(events[-1][3], max(s for _, _, s in run)))
            else:
                events.append((start, end, run[0][1], max(s for _, _, s in run)))
        run = [(t, label, score)] if t is not None and score >= THRESHOLD else []
    return events


class SoundEvents:
    def __init__(self, threads=1):
        import onnxruntime as ort

        folder = fetch_yamnet()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(str(folder / "yamnet.onnx"), sess_options=so)
        self.input = self.session.get_inputs()[0].name
        names = [line.strip() for line in (folder / "labels.txt").read_text(encoding="utf-8").splitlines()]
        self.groups = list(EVENTS)
        self.index = [[names.index(c) for c in EVENTS[g]] for g in self.groups]
        self.matrix = mel_matrix()
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WINDOW) / WINDOW)).astype(np.float32)
        self.lock = threading.Lock()

    def scores(self, patch):
        out = self.session.run(None, {self.input: patch[None, None].astype(np.float32)})[0][0]
        probs = 1.0 / (1.0 + np.exp(-out))
        return [float(probs[idx].max()) for idx in self.index]

    def detect(self, path):
        with self.lock:
            return self._detect(path)

    def _detect(self, path):
        proc = audio_stream(path, None)
        samples, frames, levels = np.zeros(0, np.float32), np.zeros((0, MELS), np.float32), np.zeros(0, np.float32)
        frame_base, patch_start, times, rows, completed = 0, 0, [], [], False
        try:
            while True:
                chunk = proc.stdout.read(READ_S * SR * 4)
                if chunk:
                    samples = np.concatenate([samples, np.frombuffer(chunk[:len(chunk) // 4 * 4], np.float32)])
                new, new_levels = log_mel(samples, self.matrix, self.window)
                if len(new):
                    frames = np.concatenate([frames, new])
                    levels = np.concatenate([levels, new_levels])
                    samples = samples[len(new) * HOP:]
                while patch_start + PATCH <= frame_base + len(frames):
                    local = patch_start - frame_base
                    if levels[local:local + PATCH].max() >= QUIET_DB:
                        rows.append(self.scores(frames[local:local + PATCH]))
                    else:
                        rows.append([0.0] * len(self.groups))
                    times.append(patch_start * HOP / SR)
                    patch_start += PATCH_HOP
                drop = patch_start - frame_base
                if drop > 0:
                    frames = frames[drop:]
                    levels = levels[drop:]
                    frame_base += drop
                if not chunk:
                    break
            completed = True
        finally:
            finish(proc, completed)
        if not rows:
            return []
        return find_events(times, np.array(rows), self.groups)
