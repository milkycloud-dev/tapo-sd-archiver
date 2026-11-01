import hashlib
import os
import subprocess
import threading
import urllib.request

import numpy as np

from backup import DATA, FFMPEG, log

SR = 16000
FRAME = 1600
FPS = SR // FRAME
PIECE_S = 20
CUT_SEARCH_S = 4
READ_S = 60
NORMALIZE = os.environ.get("ASR_FILTER", "highpass=f=80,dynaudnorm=f=150:g=15")
GATE_P90_DB = float(os.environ.get("ASR_GATE_P90_DB", "-47") or -47)
GATE_LOUD = float(os.environ.get("ASR_GATE_LOUD", "0.03") or 0.03)
PHRASE_GAP = 1.2
FILLER = set("аугдмхэоыйни")
MIN_CONTENT_LETTERS = 5
ACTIVE_DB = 6
ACTIVE_PAD = 5
VAD_MIN_S = 0.3
VAD_PAD_S = 0.5
VAD_OPTIONS = {
    "threshold": 0.3, "min_speech_duration_ms": 150, "min_silence_duration_ms": 300,
    "speech_pad_ms": 100, "max_speech_duration_s": 20,
}
MODELS = {
    "gigaam-v3-e2e-ctc": (
        "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/322c3b29492673eb7d0b434bfa9dfb8653e34d02/",
        {
            "config.json": "0641fcf73f4af791c73f05083e38a658ff5dcbee3534a0c61396c10f4b97f0fe",
            "v3_e2e_ctc.int8.onnx": "2e3fcb7a7b66030336fd10c2fcfb033bd1dc7e1bf238fe5cfd83b1d0cfc9d28e",
            "v3_e2e_ctc.yaml": "e67eca3a311ad7c8813d36dff6b8eeba7ad3459fd811d6faea2a26535754a358",
            "v3_e2e_ctc_vocab.txt": "142de7570b3de5b3035ce111a89c228e80e6085273731d944093ddf24fa539cd",
        },
    ),
    "silero-vad": (
        "https://huggingface.co/istupakov/silero-vad-onnx/resolve/b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6/",
        {
            "config.json": "1094039d370c82889582ba739a3d1caac5754c8b3a17a66a534200c9f72086e2",
            "silero_vad.onnx": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        },
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name):
    base, files = MODELS[name]
    folder = DATA / "models" / name
    folder.mkdir(parents=True, exist_ok=True)
    for file, want in files.items():
        path = folder / file
        if path.exists() and sha256(path) == want:
            continue
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(base + file, timeout=60) as resp, open(tmp, "wb") as out:
            while block := resp.read(1 << 20):
                out.write(block)
        got = sha256(tmp)
        if got != want:
            tmp.unlink()
            raise RuntimeError(f"{name}/{file}: sha256 {got} != {want}")
        tmp.replace(path)
        log(f"model file {name}/{file} verified")
    return folder


def audio_stream(path, filters):
    args = [FFMPEG, "-v", "fatal", "-i", str(path), "-vn", "-ac", "1"]
    if filters:
        args += ["-af", filters]
    return subprocess.Popen(args + ["-ar", str(SR), "-f", "f32le", "pipe:"], stdout=subprocess.PIPE)


def finish(proc, completed):
    if completed:
        proc.stdout.read()
    else:
        proc.kill()
    proc.stdout.close()
    if proc.wait() != 0 and completed:
        raise RuntimeError(f"ffmpeg audio decode failed with code {proc.returncode}")


def frame_levels(path):
    proc = audio_stream(path, None)
    levels, rest, completed = [], b"", False
    try:
        while chunk := proc.stdout.read(READ_S * SR * 4):
            data = rest + chunk
            usable = len(data) // (FRAME * 4) * FRAME * 4
            rest = data[usable:]
            frames = np.frombuffer(data[:usable], np.float32).reshape(-1, FRAME)
            levels.append(20 * np.log10(np.sqrt((frames.astype(np.float64) ** 2).mean(1)) + 1e-9))
        completed = True
    finally:
        finish(proc, completed)
    return np.concatenate(levels) if levels else np.zeros(0)


def plan_pieces(levels):
    pieces, pos, total = [], 0, len(levels)
    while pos < total:
        end = min(pos + PIECE_S * FPS, total)
        if end < total:
            low = end - CUT_SEARCH_S * FPS
            end = low + int(np.argmin(levels[low:end])) + 1
        seg = levels[pos:end]
        loud = float((seg > np.percentile(seg, 10) + 12).mean())
        keep = float(np.percentile(seg, 90)) > GATE_P90_DB or loud >= GATE_LOUD
        pieces.append((pos, end, keep))
        pos = end
    return pieces


def merge_regions(regions, length):
    out = []
    for a, b in sorted(regions):
        a, b = max(a, 0), min(b, length)
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(r) for r in out]


def active_regions(levels, length):
    if not len(levels):
        return [(0, length)]
    active = levels > np.percentile(levels, 10) + ACTIVE_DB
    if active.mean() >= 0.8 or not active.any():
        return [(0, length)]
    idx = np.flatnonzero(active)
    return merge_regions([((i - ACTIVE_PAD) * FRAME, (i + ACTIVE_PAD + 1) * FRAME) for i in idx], length)


def compact(piece, regions):
    audio = np.concatenate([piece[a:b] for a, b in regions]) if regions else piece[:0]
    starts = np.cumsum([0] + [b - a for a, b in regions])

    def to_original(t):
        sample = t * SR
        k = max(int(np.searchsorted(starts, sample, side="right")) - 1, 0)
        k = min(k, len(regions) - 1)
        return (regions[k][0] + sample - starts[k]) / SR

    return audio, to_original


def is_filler(word):
    letters = [ch.lower() for ch in word if ch.isalpha()]
    return not letters or set(letters) <= FILLER


def meaningful(words):
    content = [n for n in (sum(ch.isalpha() for ch in w) for w in words if not is_filler(w)) if n >= 2]
    return sum(content) >= MIN_CONTENT_LETTERS or len(content) >= 2


def split_phrases(tokens, timestamps, offset):
    words = []
    for token, t in zip(tokens, timestamps):
        if token.startswith(" ") or not words:
            words.append([t, t, token.strip()])
        else:
            words[-1][1] = t
            words[-1][2] += token
    words = [w for w in words if w[2]]
    if not meaningful([w[2] for w in words]):
        return []
    phrases, current = [], []
    for w in words:
        if current and (w[0] - current[-1][1] > PHRASE_GAP or current[-1][2][-1] in ".?!…"):
            phrases.append(current)
            current = []
        current.append(w)
    if current:
        phrases.append(current)
    return [(offset + p[0][0], offset + p[-1][1] + 0.3, " ".join(w[2] for w in p)) for p in phrases
            if any(sum(ch.isalpha() for ch in w[2]) >= 2 for w in p)]


class Recognizer:
    def __init__(self, threads=1):
        import onnx_asr
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        model = onnx_asr.load_model(
            "gigaam-v3-e2e-ctc", fetch("gigaam-v3-e2e-ctc"), quantization="int8", sess_options=so
        )
        self.asr = model.with_timestamps()
        vad_options = ort.SessionOptions()
        vad_options.intra_op_num_threads = 1
        vad_options.inter_op_num_threads = 1
        vad_options.enable_cpu_mem_arena = False
        self.vad = onnx_asr.load_vad("silero", fetch("silero-vad"), sess_options=vad_options)
        self.lock = threading.Lock()

    def transcribe(self, path):
        with self.lock:
            return self._transcribe(path)

    def speech_regions(self, piece):
        segments = next(iter(self.vad.segment_batch(piece[None], np.array([len(piece)]), SR, **VAD_OPTIONS)))
        segments = list(segments)
        if sum(b - a for a, b in segments) < VAD_MIN_S * SR:
            return []
        pad = int(VAD_PAD_S * SR)
        return merge_regions([(a - pad, b + pad) for a, b in segments], len(piece))

    def _transcribe(self, path):
        levels = frame_levels(path)
        pieces = plan_pieces(levels)
        proc = audio_stream(path, NORMALIZE)
        buf, buf_start, phrases, completed = np.zeros(0, np.float32), 0, [], False
        try:
            for first, last, keep in pieces:
                a, b = first * FRAME, last * FRAME
                while buf_start + len(buf) < b:
                    chunk = proc.stdout.read(READ_S * SR * 4)
                    if not chunk:
                        break
                    buf = np.concatenate([buf, np.frombuffer(chunk[:len(chunk) // 4 * 4], np.float32)])
                piece = buf[a - buf_start:b - buf_start]
                buf = buf[b - buf_start:]
                buf_start = b
                if len(piece) < SR // 2:
                    continue
                regions = active_regions(levels[first:last], len(piece)) if keep else self.speech_regions(piece)
                audio, to_original = compact(piece, regions)
                if len(audio) < SR // 4:
                    continue
                res = self.asr.recognize(audio, sample_rate=SR)
                phrases += split_phrases(res.tokens, [to_original(t) for t in res.timestamps], a / SR)
            completed = True
        finally:
            finish(proc, completed)
        return phrases
