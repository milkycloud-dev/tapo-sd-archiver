import argparse
import asyncio
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta

import archive
import backup
import control
from archive import log

LIVE_WORK = archive.DATA / "live_work"
DAY_ATTEMPTS = 3
DAY_GIVE_UP = 48
MISSING_SHOWN = 10
ROOM_POLL = 30
ACTIVE = set()
STATUS = {"state": "starting", "cameras": {}, "last_error": "", "pid": os.getpid()}


def report(state=None, camera=None, **fields):
    if STATUS.get("dry"):
        return
    if state:
        STATUS["state"] = state
    if camera:
        STATUS["cameras"].setdefault(camera, {}).update(fields)
    else:
        STATUS.update(fields)
    control.write_status(STATUS)


def clip_time(clip):
    return clip.start.replace(tzinfo=None)


def part_segments(chapters, start, duration):
    segments = []
    for begin, length, clip in chapters:
        if begin < start + duration and begin + length > start:
            shift = max(start - begin, 0.0)
            pos = max(begin - start, 0.0)
            end = min(begin + length - start, duration)
            cam = clip_time(clip) + timedelta(seconds=shift)
            segments.append({"pos": pos, "end": end, "cam": cam, "cam_end": cam + timedelta(seconds=end - pos)})
    return segments


def keyframe_times(path):
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    times = []
    for line in proc.stdout.splitlines():
        parts = line.split(",")
        if len(parts) >= 2 and parts[1].startswith("K"):
            try:
                times.append(float(parts[0]))
            except ValueError:
                pass
    return times or [0.0]


def part_note(index, total, missing):
    note = f"\nчасть {index}/{total}" if total > 1 else ""
    if missing:
        names = ", ".join(f"{clip_time(c):%H:%M:%S}" for c in missing[:MISSING_SHOWN])
        extra = f" и ещё {len(missing) - MISSING_SHOWN}" if len(missing) > MISSING_SHOWN else ""
        note += f"\nне скачано: {names}{extra}"
    return note


def camera_floor(run, camera):
    posted = [e["name"][:8] for e in run.done.values()
              if e.get("camera") == camera and e.get("messages") and not e.get("live")]
    done = [d for d, info in run.state.get("live", {}).get(camera, {}).items() if info.get("done")]
    return max(posted + done, default="")


async def fetch_day(s, dev, clips, day_dir):
    segments, missing = [], []
    for n, clip in enumerate(clips, 1):
        mp4 = day_dir / f"{clip.start:%H%M%S}.mp4"
        if not mp4.exists():
            ts = mp4.with_suffix(".ts")
            if not await backup.fetch_clip(s, dev, clip, ts):
                missing.append(clip)
                continue
            try:
                await archive.in_thread(backup.clip_to_mp4, ts, mp4)
            except RuntimeError as exc:
                log(f"  {dev.name} {clip.start:%H:%M:%S}: {exc}")
                missing.append(clip)
                continue
            finally:
                ts.unlink(missing_ok=True)
        segments.append((clip, mp4))
        report(camera=dev.name, clips=f"{n}/{len(clips)}")
        if n % 10 == 0 or n == len(clips):
            log(f"  {dev.name} {clips[0].start:%Y%m%d}: {n}/{len(clips)} clips")
    return segments, missing


def build_parts(segments, day_dir):
    runs, parts = [], []
    for clip, mp4 in segments:
        _, width, height = backup.media_info(mp4)
        if runs and runs[-1][0] == (width, height):
            runs[-1][1].append((clip, mp4))
        else:
            runs.append(((width, height), [(clip, mp4)]))
    for i, (_, run_segments) in enumerate(runs):
        out = day_dir / f"day_{i}.mp4"
        chapters = backup.concat_day(run_segments, out)
        start = 0.0
        for path, _, duration, width, height in backup.split_parts(out, chapters, backup.MTPROTO_LIMIT):
            parts.append((path, part_segments(chapters, start, duration), duration, width, height))
            start += duration
    return parts


async def post_part(run, poster, post_chat, camera, key, part, index, total, missing):
    path, segments, duration, width, height = part
    events = archive.merge_events(segments)
    start = segments[0]["cam"]
    suffix = f"_part{index}" if total > 1 else ""
    name = f"{start:%Y%m%d_%H%M%S}_live{suffix}.mp4"
    caption, rest = archive.build_caption(start, duration, events, part_note(index, total, missing))
    log(f"[{camera}] {name}: {len(events)} events, {path.stat().st_size / 1e6:.0f} MB")
    if run.args.dry_run:
        print(caption)
        for text in rest:
            print("--- reply ---\n" + text)
        return
    audio = path.with_suffix(".opus")
    await archive.in_thread(archive.extract_audio, path, audio)
    keyframes = await archive.in_thread(keyframe_times, path)
    topic = await archive.ensure_topic(poster, post_chat, run.state, camera)
    ids = await archive.upload(poster, post_chat, topic, path, caption, rest, duration, width, height, name)
    run.done[key] = {"camera": camera, "name": name, "messages": ids, "events": len(events), "live": True,
                     "posted": datetime.now().isoformat(timespec="seconds")}
    run.speech_dir.mkdir(parents=True, exist_ok=True)
    meta = archive.speech_meta({"start": start, "duration": duration, "width": width, "height": height,
                                "segments": segments, "events": events, "keyframes": keyframes}, ids[0])
    audio_path, meta_path, video_path = archive.speech_files(run, key)
    path.replace(video_path)
    audio.replace(audio_path)
    archive.save_json(meta_path, meta)
    run.save()


async def process_day(run, s, dev, camera, date, poster, post_chat):
    info = run.state.setdefault("live", {}).setdefault(camera, {}).setdefault(date, {})
    clips = sorted(await dev.search_day(date), key=lambda c: c.start)
    if not clips:
        info["done"] = True
        run.save()
        log(f"{camera} {date}: no clips")
        return True
    safe = backup.safe_name(camera)
    day_dir = LIVE_WORK / safe / date
    day_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    log(f"{camera} {date}: {len(clips)} clips, {backup.fmt_duration(sum(c.duration_seconds for c in clips))}")
    segments, missing = await fetch_day(s, dev, clips, day_dir)
    if missing:
        info["attempts"] = info.get("attempts", 0) + 1
        run.save()
        if not segments and info["attempts"] >= DAY_GIVE_UP:
            log(f"{camera} {date}: gave up, {len(clips)} clips could not be downloaded")
            info.update(done=True, lost=len(clips))
            run.save()
            shutil.rmtree(day_dir, ignore_errors=True)
            return True
        if info["attempts"] < DAY_ATTEMPTS or not segments:
            log(f"{camera} {date}: {len(missing)} clips failed, attempt {info['attempts']}, retrying later")
            return False
    parts = await archive.in_thread(build_parts, segments, day_dir)
    log(f"{camera} {date}: downloaded and joined in {time.monotonic() - t0:.0f}s, {len(parts)} part(s)")
    for index, part in enumerate(parts, 1):
        key = f"live_{safe}_{date}_{index}"
        if run.done.get(key, {}).get("messages"):
            part[0].unlink(missing_ok=True)
            continue
        await post_part(run, poster, post_chat, camera, key, part, index, len(parts), missing)
    if run.args.dry_run:
        return True
    info.update(done=True, parts=len(parts), missing=len(missing))
    run.save()
    shutil.rmtree(day_dir, ignore_errors=True)
    log(f"{camera} {date}: posted")
    report(camera=camera, last_posted=date, clips="", missing_last=len(missing))
    return True


def blocked_reason(run):
    settings = control.load_settings()
    archive.MEMORY_GATE = settings["memory_gate_pct"] / 100
    if settings["paused"] and not run.args.dry_run:
        return "paused"
    if archive.free_gb() < settings["min_free_gb"] or archive.dir_gb(run.speech_dir) > settings["queue_gb"]:
        return "waiting"
    return ""


async def wait_for_room(run):
    warned = False
    while reason := blocked_reason(run):
        report(reason)
        if not warned:
            log(f"  {reason}: free {archive.free_gb():.1f} GB, speech queue {archive.dir_gb(run.speech_dir):.1f} GB")
            warned = True
        await asyncio.sleep(ROOM_POLL)
    report("running")


async def camera_pass(run, s, cam, poster, post_chat):
    dev = s.device(cam.device_id, name=cam.name)
    try:
        today = await dev.camera_today()
        floor = camera_floor(run, cam.name)
        dates = sorted(d for d in await dev.recording_dates("20200101", today) if floor < d < today)
        if run.args.date:
            dates = [d for d in dates if d == run.args.date]
        log(f"{cam.name}: {len(dates)} days pending after {floor}")
        report(camera=cam.name, pending=len(dates), today=today, floor=floor)
        for done, date in enumerate(dates):
            await wait_for_room(run)
            report(camera=cam.name, date=date, pending=len(dates) - done, clips="")
            if not await process_day(run, s, dev, cam.name, date, poster, post_chat):
                report(camera=cam.name, error=f"{date}: retrying")
                return False
            report(camera=cam.name, pending=len(dates) - done - 1, error="")
        report(camera=cam.name, date="", clips="")
        return True
    except Exception as exc:
        log(f"{cam.name}: pass failed: {type(exc).__name__}: {exc}")
        report(camera=cam.name, error=f"{type(exc).__name__}: {exc}"[:200])
        return False


def session_mtime():
    return backup.SESSION_FILE.stat().st_mtime if backup.SESSION_FILE.exists() else 0


async def idle(delay):
    wake, session = control.read_control().get("wake", 0), session_mtime()
    deadline = time.monotonic() + delay
    while time.monotonic() < deadline:
        await asyncio.sleep(min(5, max(deadline - time.monotonic(), 0)))
        if control.read_control().get("wake", 0) != wake or session_mtime() != session:
            return


def login_needed(exc):
    if STATUS.get("login") != "required":
        log(f"login required: {exc}")
    report("login_required", login="required", last_error="login required")


async def camera_loop(run, cam, poster, post_chat):
    while True:
        ok = login = False
        ACTIVE.add(cam.name)
        try:
            s = await backup.open_saved_session()
            report("running", login="ok", last_error="")
            async with s:
                s.app_server = cam.app_server or s.app_server
                ok = await camera_pass(run, s, cam, poster, post_chat)
        except backup.LoginRequired as exc:
            login = True
            login_needed(exc)
        except Exception as exc:
            log(f"{cam.name}: pass failed: {type(exc).__name__}: {exc}")
            report(camera=cam.name, error=f"{type(exc).__name__}: {exc}"[:200])
        ACTIVE.discard(cam.name)
        if run.args.once:
            return ok
        settings = control.load_settings()
        delay = (settings["interval_min"] if ok else settings["retry_min"]) * 60
        if not login:
            log(f"{cam.name}: next check in {delay}s")
            report(None if ACTIVE else "sleeping" if ok else "retrying", next_check=time.time() + delay)
        await idle(delay)


async def main_loop(run, poster, post_chat, stop):
    while True:
        try:
            s = await backup.open_saved_session()
            report(login="ok", last_error="")
            async with s:
                cams = [c for c in await s.cameras() if run.args.camera.lower() in c.name.lower()]
            break
        except backup.LoginRequired as exc:
            login_needed(exc)
        except Exception as exc:
            log(f"live pass failed: {type(exc).__name__}: {exc}")
            report(last_error=f"{type(exc).__name__}: {exc}"[:200])
        if run.args.once:
            stop.set()
            return
        await idle(control.load_settings()["retry_min"] * 60)
    await asyncio.gather(*(camera_loop(run, c, poster, post_chat) for c in cams))
    stop.set()


async def speech_loop(run, poster, post_chat, recognizer, detector, stop):
    work = archive.WORK / "live_speech"
    work.mkdir(parents=True, exist_ok=True)
    while True:
        pending = sorted((e["name"], k) for k, e in run.done.items() if e.get("live") and "speech" not in e)
        report(speech_pending=len(pending), speech_file=pending[0][0] if pending else "")
        if not pending:
            if stop.is_set():
                return
            await asyncio.sleep(10)
            continue
        name, key = pending[0]
        entry = run.done[key]
        files = archive.speech_files(run, key)
        if not all(p.exists() for p in files):
            log(f"speech {name}: queue files missing, skipped")
            entry["speech"] = []
            entry["speech_skipped"] = "queue files missing"
            run.save()
            continue
        item = {"id": key, "name": name, "camera": entry["camera"]}
        try:
            await archive.speech_item(run, item, entry, poster, post_chat, None, None, recognizer, detector, None, work)
        except Exception as exc:
            failures = run.state.setdefault("speech_failures", {})
            failures[key] = failures.get(key, 0) + 1
            log(f"speech {name} failed ({failures[key]}/{archive.SPEECH_MAX_FAILURES}): {type(exc).__name__}: {exc}")
            if failures[key] >= archive.SPEECH_MAX_FAILURES:
                entry["speech"] = []
                entry["speech_skipped"] = f"{type(exc).__name__}: {exc}"[:300]
                for path in files:
                    path.unlink(missing_ok=True)
            run.save()
            await asyncio.sleep(archive.SPEECH_RETRY_DELAY * failures[key])


async def heartbeat(stop):
    started = control.read_control().get("restart", 0)
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), 15)
        except TimeoutError:
            pass
        report()
        if control.read_control().get("restart", 0) != started:
            log("restart requested from the panel")
            os.kill(os.getpid(), signal.SIGTERM)
            return


async def cmd_live(args):
    os.nice(10)
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    run = archive.Run(args, [])
    stop = asyncio.Event()
    LIVE_WORK.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        STATUS["dry"] = True
        try:
            await main_loop(run, None, None, stop)
        finally:
            await archive.drain_threads()
            shutil.rmtree(LIVE_WORK, ignore_errors=True)
        return
    archive.clear_work()
    archive.WORK.mkdir(parents=True, exist_ok=True)
    archive.SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    poster, post_chat = await archive.connect("post")
    try:
        recognizer = await asyncio.to_thread(archive.Recognizer, archive.ASR_THREADS)
        detector = await asyncio.to_thread(archive.SoundEvents, 1)
        async with asyncio.TaskGroup() as group:
            group.create_task(main_loop(run, poster, post_chat, stop))
            group.create_task(speech_loop(run, poster, post_chat, recognizer, detector, stop))
            group.create_task(heartbeat(stop))
    finally:
        await archive.drain_threads()
        await poster.disconnect()
        archive.clear_work()
        log(f"stopped, free {archive.free_gb():.1f} GB, speech queue {archive.dir_gb(archive.SPEECH_DIR):.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        args.once = True
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(cmd_live(args))
    except (asyncio.CancelledError, KeyboardInterrupt):
        sys.exit(143)


if __name__ == "__main__":
    main()
