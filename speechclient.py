import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


class SpeechClient:
    def __init__(self, url, key, poll_s=3.0, retry_s=30.0):
        self.url = url.rstrip("/")
        self.key = key
        self.poll_s = poll_s
        self.retry_s = retry_s
        self.last = None

    def _req(self, method, path, data=None, timeout=60):
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={"X-Client-Key": self.key})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def status(self):
        try:
            self.last = self._req("GET", "/v1/status", timeout=15)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self.last = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        return self.last

    def _wait(self, on_wait, why):
        if on_wait:
            on_wait(why)
        time.sleep(self.retry_s)

    def _submit(self, data, query, on_wait):
        while True:
            try:
                return self._req("POST", "/v1/jobs?" + query, data, timeout=600)["key"]
            except urllib.error.HTTPError as exc:
                if exc.code == 429 or exc.code >= 500:
                    self._wait(on_wait, f"worker busy ({exc.code})")
                    continue
                raise RuntimeError(f"worker {exc.code}: {exc.read()[:300].decode(errors='replace')}")
            except (urllib.error.URLError, OSError) as exc:
                self._wait(on_wait, f"worker unreachable: {type(exc).__name__}")

    def transcribe(self, path, language="ru", sounds=True, events=False, name="", on_wait=None):
        with open(path, "rb") as f:
            data = f.read()
        query = urllib.parse.urlencode({"ext": os.path.splitext(path)[1].lower() or ".ogg", "language": language,
                                        "sounds": int(bool(sounds)), "events": int(bool(events)), "name": name[:200]})
        key = self._submit(data, query, on_wait)
        while True:
            try:
                job = self._req("GET", f"/v1/jobs/{key}", timeout=30)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    key = self._submit(data, query, on_wait)
                    continue
                self._wait(on_wait, f"worker answered {exc.code}")
                continue
            except (urllib.error.URLError, OSError) as exc:
                self._wait(on_wait, f"worker unreachable: {type(exc).__name__}")
                continue
            if job.get("state") in ("done", "failed"):
                try:
                    self._req("POST", f"/v1/jobs/{key}/ack", b"", timeout=30)
                except (urllib.error.URLError, OSError):
                    pass
                if job["state"] == "failed":
                    raise RuntimeError(job.get("error") or "worker job failed")
                if on_wait:
                    on_wait(None)
                return job
            time.sleep(self.poll_s)
