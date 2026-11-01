import argparse
import asyncio
import json
import os
import re
import socket
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

from ontapo import Clip, OnTapo, OnTapoError

ROOT = Path(__file__).resolve().parent
SESSION_FILE = ROOT / ".tapo_session.json"
OUT_DIR = ROOT / "recordings" / "probe"
CONTACTED = set()
_getaddrinfo = socket.getaddrinfo


def logged_getaddrinfo(host, *args, **kwargs):
    CONTACTED.add(host.decode() if isinstance(host, bytes) else str(host))
    return _getaddrinfo(host, *args, **kwargs)


socket.getaddrinfo = logged_getaddrinfo


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def save_session(s):
    SESSION_FILE.write_text(json.dumps(s.to_dict()), encoding="utf-8")
    try:
        os.chmod(SESSION_FILE, 0o600)
    except OSError:
        pass


async def open_session():
    if SESSION_FILE.exists():
        s = OnTapo.from_dict(json.loads(SESSION_FILE.read_text(encoding="utf-8")))
        try:
            await s.cameras()
            print("session: reused saved token")
            return s
        except OnTapoError:
            try:
                await s.refresh()
                await s.cameras()
                save_session(s)
                print("session: refreshed saved token")
                return s
            except OnTapoError:
                await s.aclose()
    email = os.environ.get("TAPO_EMAIL")
    password = os.environ.get("TAPO_PASSWORD")
    if not email or not password:
        sys.exit("TAPO_EMAIL / TAPO_PASSWORD missing in .env")
    s = await OnTapo.login(email, password, mfa=lambda: input("MFA code from email: ").strip())
    save_session(s)
    print("session: logged in")
    return s


def ts_valid(path):
    data = path.read_bytes()[: 188 * 50]
    if len(data) < 188:
        return False
    return all(data[i] == 0x47 for i in range(0, len(data) - 187, 188))


async def step(label, coro):
    try:
        return await coro
    except Exception as exc:
        print(f"  {label}: FAILED {type(exc).__name__}: {exc}")
        return None


async def probe_camera(s, cam, args):
    print(f"\n=== {cam.name} | {cam.model} | {cam.device_id}")
    if cam.app_server and cam.app_server != s.app_server:
        s.app_server = cam.app_server
    dev = s.device(cam.device_id, name=cam.name)

    summary = await step("summary", dev.summary())
    if summary:
        print("  " + summary.describe().replace("\n", "\n  "))

    sd = await step("sd_card_status", dev.sd_card_status())
    if sd:
        print(f"  sd: {sd.summary}; recording since {sd.recording_since}")

    dates = await step("recording_dates", dev.recent_recording_dates(args.days))
    if not dates:
        print("  no recording dates found")
        return
    dates = sorted(dates)
    print(f"  dates with recordings (last {args.days}d): {', '.join(dates)}")

    clips = await step("search_day", dev.search_day(dates[-1]))
    if not clips:
        print(f"  no clips for {dates[-1]}")
        return
    clips.sort(key=lambda c: c.start)
    total = sum(c.duration_seconds for c in clips)
    types = Counter(c.video_type for c in clips)
    print(f"  {dates[-1]}: {len(clips)} clips, {total / 3600:.2f} h total, types {dict(types)}")
    print(f"  first {clips[0].start:%H:%M:%S}-{clips[0].end:%H:%M:%S}, last {clips[-1].start:%H:%M:%S}-{clips[-1].end:%H:%M:%S}")

    if args.no_download:
        return
    now = await dev.camera_now()
    done = [c for c in clips if c.end < now - timedelta(minutes=2)] or clips
    src = done[-1]
    seg = Clip(src.start, min(src.end, src.start + timedelta(seconds=args.seconds)), src.video_type)
    secs = seg.duration_seconds
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w-]+", "_", cam.name) or cam.device_id
    out = OUT_DIR / f"{safe}_{seg.start:%Y%m%d_%H%M%S}_{int(secs)}s.ts"
    print(f"  downloading {seg.start:%Y-%m-%d %H:%M:%S} +{secs:.0f}s ...")
    t0 = time.monotonic()
    n = await step("download", dev.download(seg, str(out)))
    dt = time.monotonic() - t0
    if not n:
        return
    kbps = n * 8 / secs / 1000
    print(f"  saved {out.relative_to(ROOT)}: {n / 1e6:.1f} MB in {dt:.1f}s")
    print(f"  speed {secs / dt:.2f}x realtime, {n / dt / 1e6:.2f} MB/s")
    print(f"  bitrate ~{kbps:.0f} kbps, ~{kbps * 86400 / 8 / 1e6:.1f} GB/day of footage")
    print(f"  mpeg-ts sync check: {'OK' if ts_valid(out) else 'BAD'}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--camera", default="")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    load_env(ROOT / ".env")
    try:
        s = await open_session()
        async with s:
            cams = await s.cameras()
            print(f"cameras on account: {len(cams)}")
            for cam in cams:
                print(f"  - {cam.name} ({cam.model}) server={cam.app_server}")
            for cam in cams:
                if args.camera.lower() in cam.name.lower():
                    await probe_camera(s, cam, args)
            save_session(s)
    finally:
        print(f"\nhosts contacted: {', '.join(sorted(CONTACTED)) or 'none'}")


if __name__ == "__main__":
    asyncio.run(main())
