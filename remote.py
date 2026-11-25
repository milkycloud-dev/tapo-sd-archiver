import os
import threading

from backup import log
from speechclient import SpeechClient

LANGUAGE = os.environ.get("SPEECH_LANGUAGE", "ru") or "ru"


class RemoteSpeech:
    def __init__(self, url, key):
        self.client = SpeechClient(url, key)
        self.lock = threading.Lock()
        self.cache = {}
        self.waiting = None

    def on_wait(self, why):
        if why != self.waiting:
            log(f"  speech worker: {why}" if why else "  speech worker: back")
        self.waiting = why

    def job(self, path):
        key = (str(path), os.path.getsize(path), os.path.getmtime(path))
        with self.lock:
            if key not in self.cache:
                job = self.client.transcribe(str(path), LANGUAGE, sounds=False, events=True,
                                             name=os.path.basename(str(path)), on_wait=self.on_wait)
                self.cache = {key: job}
            return self.cache[key]

    def transcribe(self, path):
        return [(float(a), float(b), text) for a, b, text in self.job(path).get("phrases") or []]

    def detect(self, path):
        return [(float(a), float(b), label, 1.0) for a, b, label in self.job(path).get("events") or []]


_shared = None
_shared_lock = threading.Lock()


def shared():
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = RemoteSpeech(os.environ["SPEECH_URL"], os.environ.get("SPEECH_KEY", ""))
        return _shared


def Recognizer(threads=1):
    return shared()


def SoundEvents(threads=1):
    return shared()
