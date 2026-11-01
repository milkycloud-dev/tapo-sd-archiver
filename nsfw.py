import hashlib
import os
import subprocess
import threading
import urllib.request

import numpy as np

from backup import DATA, FFMPEG, log

MODEL_URL = ("https://huggingface.co/onnx-community/nsfw_image_detection-ONNX/resolve/"
             "1ceb3c7fe1e9f3f2507e6df577437f23a9149fd5/onnx/model_quantized.onnx")
MODEL_SHA256 = "d9afb1e057104e6cc8616d174f0f6a8b8b0389c839eea7d5efa4bdf1a77efd27"
MODEL_PATH = DATA / "models" / "nsfw-falconsai-int8.onnx"
SIZE = 224
STEP_S = float(os.environ.get("NSFW_STEP_S", "45") or 45)
THRESHOLD = float(os.environ.get("NSFW_THRESHOLD", "0.8") or 0.8)
STRONG = float(os.environ.get("NSFW_STRONG", "0.9") or 0.9)
MIN_FRAMES = 2
BATCH = 16


def fetch_model():
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(MODEL_URL, timeout=120) as resp:
        data = resp.read()
    got = hashlib.sha256(data).hexdigest()
    if got != MODEL_SHA256:
        raise RuntimeError(f"nsfw model: sha256 {got} != {MODEL_SHA256}")
    tmp = MODEL_PATH.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(MODEL_PATH)
    log("model nsfw falconsai int8 verified")
    return MODEL_PATH


def sample_frames(path, step=STEP_S):
    proc = subprocess.run(
        [FFMPEG, "-v", "fatal", "-skip_frame", "nokey", "-i", str(path), "-an",
         "-vf", f"fps=1/{step},scale={SIZE}:{SIZE}:flags=bilinear,format=rgb24",
         "-f", "rawvideo", "pipe:"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nsfw frame extraction failed with code {proc.returncode}")
    return np.frombuffer(proc.stdout, np.uint8).reshape(-1, SIZE, SIZE, 3)


def preprocess(frames):
    return ((frames.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(0, 3, 1, 2)


def verdict(probs):
    hits = int((probs >= THRESHOLD).sum())
    top = float(probs.max()) if len(probs) else 0.0
    return hits >= MIN_FRAMES or top >= STRONG, top, hits


class NsfwDetector:
    def __init__(self, threads=1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(str(fetch_model()), sess_options=so)
        self.input = self.session.get_inputs()[0].name
        self.lock = threading.Lock()

    def probabilities(self, frames):
        out = []
        for b in range(0, len(frames), BATCH):
            logits = self.session.run(None, {self.input: preprocess(frames[b:b + BATCH])})[0]
            shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
            out.append(shifted[:, 1] / shifted.sum(axis=1))
        return np.concatenate(out) if out else np.zeros(0)

    def check(self, path):
        frames = sample_frames(path)
        with self.lock:
            probs = self.probabilities(frames)
        return verdict(probs)
