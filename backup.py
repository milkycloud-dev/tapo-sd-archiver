import argparse
import asyncio
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import imageio_ffmpeg
from ontapo import OnTapo, OnTapoError, Unauthorized
from ontapo.tsdemux import extract_g711_audio

import control

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TAPO_DATA") or ROOT)
SESSION_FILE = DATA / ".tapo_session.json"
STATE_FILE = DATA / "state.json"
WORK_DIR = DATA / "work"
OUT_DIR = DATA / "recordings"
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
MTPROTO_LIMIT = 2000 * 1024 * 1024
BOTAPI_LIMIT = 50 * 1000 * 1000
CAPTION_LIMIT = 1024
MESSAGE_LIMIT = 4000
DAY_ATTEMPTS = 3


class LoginRequired(Exception):
    pass


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    source = os.environ.get("TAPO_LOG_SOURCE")
    if source:
        control.append_log(source, line)


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
    control.write_json(SESSION_FILE, s.to_dict(), mode=0o600)


async def open_saved_session():
    saved = control.read_json(SESSION_FILE, None)
    if not saved or not saved.get("token"):
        raise LoginRequired("no saved session")
    s = OnTapo.from_dict(saved)
    try:
        await s.cameras()
        return s
    except OnTapoError:
        pass
    try:
        with control.session_lock():
            current = control.read_json(SESSION_FILE, saved)
            if current.get("refreshToken") != s.refresh_token:
                await s.aclose()
                s = OnTapo.from_dict(current)
            else:
                await s.refresh()
                save_session(s)
        await s.cameras()
        return s
    except OnTapoError as exc:
        await s.aclose()
        raise LoginRequired(type(exc).__name__) from None


async def open_session():
    if SESSION_FILE.exists():
        s = OnTapo.from_dict(json.loads(SESSION_FILE.read_text(encoding="utf-8")))
        try:
            await s.cameras()
            return s
        except OnTapoError:
            try:
                await s.refresh()
                await s.cameras()
                save_session(s)
                return s
            except OnTapoError:
                await s.aclose()
    email, password = control.load_credentials()
    if not email or not password:
        sys.exit("Tapo session expired and TAPO_EMAIL / TAPO_PASSWORD missing in .env")
    s = await OnTapo.login(email, password, mfa=lambda: input("MFA code from email: ").strip())
    save_session(s)
    return s


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(STATE_FILE)


def safe_name(name):
    return re.sub(r"[^\w-]+", "_", name).strip("_") or "camera"


def run_ffmpeg(*args):
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        capture_output=True, text=True, errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-500:]}")


def media_info(path):
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True, errors="replace"
    )
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0
    v = re.search(r"Video: [^\n]*?, (\d{2,5})x(\d{2,5})", proc.stderr)
    width, height = (int(v[1]), int(v[2])) if v else (0, 0)
    return duration, width, height


def fmt_offset(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"


def fmt_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h} ч {m} мин" if h else f"{m} мин {s} с"


def video_mode():
    return "hevc" if os.environ.get("VIDEO_CODEC", "copy").strip().lower() == "hevc" else "h264"


def video_args(source_kbps):
    if video_mode() != "hevc":
        return ["-c:v", "copy"]
    encoder = os.environ.get("HEVC_ENCODER", "libx265").strip()
    args = ["-c:v", encoder, "-tag:v", "hvc1", "-pix_fmt", "yuv420p"]
    if encoder == "libx265":
        params = os.environ.get("HEVC_X265_PARAMS", "aq-mode=3:no-sao=1").strip()
        args += [
            "-preset", os.environ.get("HEVC_PRESET", "medium").strip(),
            "-crf", os.environ.get("HEVC_CRF", "24").strip(),
            "-x265-params", "log-level=error" + (":" + params if params else ""),
        ]
    ratio = float(os.environ.get("HEVC_MAX_RATIO", "0.8") or 0)
    if ratio > 0 and source_kbps > 0:
        cap = max(int(source_kbps * ratio), 100)
        args += ["-maxrate", f"{cap}k", "-bufsize", f"{cap * 2}k"]
    return args + shlex.split(os.environ.get("HEVC_EXTRA_ARGS", ""))


def clip_to_mp4(ts_path, mp4_path):
    data = ts_path.read_bytes()
    raw, fmt, rate = extract_g711_audio(data)
    duration, _, _ = media_info(ts_path) if video_mode() == "hevc" else (0.0, 0, 0)
    source_kbps = (len(data) - len(raw)) * 8 / duration / 1000 if duration else 0
    audio_path = ts_path.with_suffix(".g711")
    if raw:
        audio_path.write_bytes(raw)
        audio_in = ["-f", fmt, "-ar", rate, "-ac", 1, "-i", audio_path]
        shortest = []
    else:
        audio_in = ["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono"]
        shortest = ["-shortest"]
    tmp = mp4_path.with_name(mp4_path.stem + ".tmp.mp4")
    run_ffmpeg(
        "-fflags", "+genpts", "-i", ts_path, *audio_in,
        "-map", "0:v:0", "-map", "1:a:0", *video_args(source_kbps),
        "-c:a", "aac", "-b:a", "48k", "-ar", 16000, "-ac", 1, *shortest,
        "-avoid_negative_ts", "make_zero", "-video_track_timescale", 90000,
        "-movflags", "+faststart", tmp,
    )
    tmp.replace(mp4_path)
    audio_path.unlink(missing_ok=True)


def concat_day(segments, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    chapters = []
    offset = 0.0
    entries = []
    for clip, mp4 in segments:
        duration, _, _ = media_info(mp4)
        chapters.append((offset, duration, clip))
        offset += duration
        entries.append("file '" + mp4.resolve().as_posix().replace("'", "'\\''") + "'")
    list_file = out.with_suffix(".txt")
    list_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
    meta = [";FFMETADATA1"]
    for start, duration, clip in chapters:
        meta += [
            "[CHAPTER]", "TIMEBASE=1/1000",
            f"START={int(start * 1000)}", f"END={int((start + duration) * 1000)}",
            f"title={clip.start:%H:%M:%S}",
        ]
    meta_file = out.with_suffix(".ffmeta")
    meta_file.write_text("\n".join(meta) + "\n", encoding="utf-8")
    tmp = out.with_name(out.stem + ".tmp.mp4")
    run_ffmpeg(
        "-f", "concat", "-safe", 0, "-i", list_file, "-i", meta_file,
        "-map", "0", "-map_metadata", 1, "-map_chapters", 1,
        "-c", "copy", "-movflags", "+faststart", tmp,
    )
    tmp.replace(out)
    list_file.unlink()
    meta_file.unlink()
    return chapters


def split_parts(out, chapters, limit):
    total, width, height = media_info(out)
    if out.stat().st_size <= limit:
        return [(out, chapters, total, width, height)]
    count = math.ceil(out.stat().st_size / (limit * 0.9))
    while True:
        for old in out.parent.glob(f"{out.stem}_part*.mp4"):
            old.unlink()
        run_ffmpeg(
            "-i", out, "-map", "0", "-c", "copy", "-f", "segment",
            "-segment_time", f"{total / count:.0f}", "-reset_timestamps", 1,
            "-segment_format_options", "movflags=+faststart",
            out.with_name(f"{out.stem}_part%02d.mp4"),
        )
        paths = sorted(out.parent.glob(f"{out.stem}_part*.mp4"))
        if all(p.stat().st_size <= limit for p in paths):
            break
        count += 1
    parts = []
    start = 0.0
    for path in paths:
        duration, width, height = media_info(path)
        rel = [
            (max(c - start, 0.0), d, clip) for c, d, clip in chapters
            if c < start + duration and c + d > start + 1
        ]
        parts.append((path, rel, duration, width, height))
        start += duration
    out.unlink()
    return parts


def build_texts(header, chapters):
    lines = [f"{fmt_offset(start)} — {clip.start:%H:%M:%S}" for start, _, clip in chapters]
    caption = header
    i = 0
    while i < len(lines) and len(caption) + 1 + len(lines[i]) <= CAPTION_LIMIT:
        caption += "\n" + lines[i]
        i += 1
    rest = []
    chunk = ""
    for line in lines[i:]:
        if chunk and len(chunk) + 1 + len(line) > MESSAGE_LIMIT:
            rest.append(chunk)
            chunk = ""
        chunk += ("\n" if chunk else "") + line
    if chunk:
        rest.append(chunk)
    return caption, rest


async def bot_api(token, method, data, files=None):
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30)) as client:
            resp = await client.post(f"https://api.telegram.org/bot{token}/{method}", data=data, files=files)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Bot API {method}: {type(exc).__name__}") from None
    body = resp.json()
    if not body.get("ok"):
        raise RuntimeError(f"Bot API {method}: {body.get('description')}")
    return body["result"]


async def telegram_client():
    token = os.environ.get("TG_BOT_TOKEN", "").strip()
    chat = os.environ.get("TG_CHAT", "").strip()
    if not (token and chat):
        sys.exit("TG_BOT_TOKEN / TG_CHAT missing in .env")
    api_id = os.environ.get("TG_API_ID", "").strip()
    api_hash = os.environ.get("TG_API_HASH", "").strip()
    if api_id and api_hash:
        from telethon import TelegramClient

        client = TelegramClient(str(DATA / "bot"), int(api_id), api_hash, flood_sleep_threshold=600)
        await client.start(bot_token=token)
        entity = await client.get_input_entity(int(chat) if re.fullmatch(r"-?\d+", chat) else chat)
        return {"client": client, "entity": entity, "limit": MTPROTO_LIMIT}
    info = await bot_api(token, "getChat", {"chat_id": chat})
    log(f"Telegram: Bot API mode (no TG_API_ID), parts up to 50 MB, chat '{info.get('title')}'")
    return {"token": token, "chat": chat, "limit": BOTAPI_LIMIT}


async def tg_close(tg):
    if tg and "client" in tg:
        await tg["client"].disconnect()


async def tg_send_video(tg, path, caption, thumb, duration, width, height, progress):
    if "client" in tg:
        from telethon.tl.types import DocumentAttributeVideo

        msg = await tg["client"].send_file(
            tg["entity"], str(path), caption=caption, parse_mode=None, thumb=str(thumb),
            supports_streaming=True, progress_callback=progress,
            attributes=[DocumentAttributeVideo(
                duration=duration, w=width, h=height, supports_streaming=True
            )],
        )
        return msg.id
    result = await bot_api(
        tg["token"], "sendVideo",
        {
            "chat_id": tg["chat"], "caption": caption, "duration": int(duration),
            "width": width, "height": height, "supports_streaming": "true",
        },
        {
            "video": (path.name, path.read_bytes(), "video/mp4"),
            "thumbnail": (thumb.name, thumb.read_bytes(), "image/jpeg"),
        },
    )
    return result["message_id"]


async def tg_send_text(tg, text, reply_to):
    if "client" in tg:
        msg = await tg["client"].send_message(tg["entity"], text, reply_to=reply_to, parse_mode=None)
        return msg.id
    result = await bot_api(
        tg["token"], "sendMessage",
        {"chat_id": tg["chat"], "text": text, "reply_parameters": json.dumps({"message_id": reply_to})},
    )
    return result["message_id"]


async def upload_parts(tg, cam_name, date, parts, clip_count, missing):
    day = datetime.strptime(date, "%Y%m%d")
    ids = []
    for index, (path, chapters, duration, width, height) in enumerate(parts, 1):
        header = f"{cam_name} · {day:%d.%m.%Y}"
        if len(parts) > 1:
            header += f" · часть {index}/{len(parts)}"
        header += f"\n{clip_count} записей, {fmt_duration(sum(p[2] for p in parts))}"
        if missing:
            header += f"\nне скачано: {', '.join(f'{c.start:%H:%M:%S}' for c in missing)}"
        header += "\nвремя в ролике — время камеры:"
        caption, rest = build_texts(header, chapters)
        thumb = path.with_suffix(".jpg")
        await asyncio.to_thread(
            run_ffmpeg, "-ss", 1, "-i", path, "-frames:v", 1, "-vf", "scale=320:-2", thumb
        )
        last = [0]

        def progress(sent, total, name=path.name):
            pct = int(sent * 100 / total) if total else 100
            if pct >= last[0] + 20:
                last[0] = pct
                log(f"  upload {name}: {pct}%")

        msg_id = await tg_send_video(tg, path, caption, thumb, duration, width, height, progress)
        ids.append(msg_id)
        for text in rest:
            ids.append(await tg_send_text(tg, text, msg_id))
        thumb.unlink(missing_ok=True)
    return ids


async def fetch_clip(s, dev, clip, path):
    if path.exists():
        return True
    part = path.with_suffix(".part")
    for attempt in range(3):
        try:
            if await dev.download(clip, str(part)) > 0:
                part.replace(path)
                return True
        except Unauthorized:
            with control.session_lock():
                await s.refresh()
                save_session(s)
        except Exception as exc:
            log(f"  {dev.name} {clip.start:%H:%M:%S}: {type(exc).__name__}: {exc}")
        await asyncio.sleep(10 * (attempt + 1))
    part.unlink(missing_ok=True)
    return False


async def process_day(s, cam, date, state, tg, *, upload=True, max_clips=0):
    dev = s.device(cam.device_id, name=cam.name)
    entry = state.setdefault(cam.device_id, {}).setdefault(date, {})
    clips = sorted(await dev.search_day(date), key=lambda c: c.start)
    if max_clips:
        clips = clips[:max_clips]
    if not clips:
        log(f"{cam.name} {date}: no clips")
        entry["uploaded"] = upload
        save_state(state)
        return
    total = sum(c.duration_seconds for c in clips)
    log(f"{cam.name} {date}: {len(clips)} clips, {fmt_duration(total)}")
    name = safe_name(cam.name)
    day_dir = WORK_DIR / name / date
    day_dir.mkdir(parents=True, exist_ok=True)
    fetched, missing = [], []
    for n, clip in enumerate(clips, 1):
        ts = day_dir / f"{clip.start:%H%M%S}.ts"
        if await fetch_clip(s, dev, clip, ts):
            fetched.append((clip, ts))
        else:
            missing.append(clip)
        if n % 10 == 0 or n == len(clips):
            log(f"  {cam.name} {date}: {n}/{len(clips)} downloaded")
    if missing:
        entry["attempts"] = entry.get("attempts", 0) + 1
        save_state(state)
        if entry["attempts"] < DAY_ATTEMPTS or not fetched:
            log(f"{cam.name} {date}: {len(missing)} clips failed, retry next pass")
            return
    segments = []
    for clip, ts in fetched:
        mp4 = ts.with_suffix(f".{video_mode()}.mp4")
        if not mp4.exists():
            await asyncio.to_thread(clip_to_mp4, ts, mp4)
        segments.append((clip, mp4))
    out = OUT_DIR / name / f"{name}_{date}.mp4"
    chapters = await asyncio.to_thread(concat_day, segments, out)
    limit = tg["limit"] if upload else MTPROTO_LIMIT
    parts = await asyncio.to_thread(split_parts, out, chapters, limit)
    for path, _, duration, _, _ in parts:
        log(f"{cam.name} {date}: {path.relative_to(DATA)} {path.stat().st_size / 1e6:.1f} MB, {fmt_duration(duration)}")
    if not upload:
        return
    entry["messages"] = await upload_parts(tg, cam.name, date, parts, len(clips), missing)
    entry["uploaded"] = True
    entry["missing"] = len(missing)
    save_state(state)
    log(f"{cam.name} {date}: uploaded")
    if os.environ.get("KEEP_LOCAL") != "1":
        shutil.rmtree(day_dir, ignore_errors=True)
        for path, *_ in parts:
            path.unlink(missing_ok=True)


async def backlog(s, cam, state, tg, args):
    dev = s.device(cam.device_id, name=cam.name)
    today = await dev.camera_today()
    since = args.since or (datetime.strptime(today, "%Y%m%d") - timedelta(days=args.lookback)).strftime("%Y%m%d")
    dates = sorted(d for d in await dev.recording_dates(since, today) if since <= d < today)
    pending = [d for d in dates if not state.get(cam.device_id, {}).get(d, {}).get("uploaded")]
    log(f"{cam.name}: {len(pending)} days pending")
    for date in pending:
        try:
            await process_day(s, cam, date, state, tg)
        except Exception as exc:
            log(f"{cam.name} {date}: FAILED {type(exc).__name__}: {exc}")


async def cmd_run(args):
    state = load_state()
    tg = await telegram_client()
    try:
        while True:
            try:
                s = await open_session()
                async with s:
                    cams = [c for c in await s.cameras() if args.camera.lower() in c.name.lower()]
                    servers = {c.app_server for c in cams}
                    if len(servers) == 1:
                        s.app_server = servers.pop() or s.app_server
                        await asyncio.gather(*(backlog(s, c, state, tg, args) for c in cams))
                    else:
                        for c in cams:
                            s.app_server = c.app_server or s.app_server
                            await backlog(s, c, state, tg, args)
                    save_session(s)
            except Exception as exc:
                log(f"pass failed: {type(exc).__name__}: {exc}")
            if args.once:
                break
            log(f"sleeping {args.interval}s")
            await asyncio.sleep(args.interval)
    finally:
        await tg_close(tg)


async def cmd_day(args):
    state = load_state()
    tg = None if args.no_upload else await telegram_client()
    try:
        s = await open_session()
        async with s:
            cams = [c for c in await s.cameras() if args.camera.lower() in c.name.lower()]
            if len(cams) != 1:
                sys.exit(f"--camera must match exactly one camera, matched {len(cams)}")
            cam = cams[0]
            s.app_server = cam.app_server or s.app_server
            await process_day(
                s, cam, args.date, state, tg, upload=not args.no_upload, max_clips=args.max_clips
            )
            save_session(s)
    finally:
        await tg_close(tg)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--camera", default="")
    run.add_argument("--since", default="")
    run.add_argument("--lookback", type=int, default=120)
    run.add_argument("--interval", type=int, default=3600)
    run.add_argument("--once", action="store_true")
    day = sub.add_parser("day")
    day.add_argument("--camera", required=True)
    day.add_argument("--date", required=True)
    day.add_argument("--no-upload", action="store_true")
    day.add_argument("--max-clips", type=int, default=0)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env(ROOT / ".env")
    asyncio.run(cmd_run(args) if args.cmd == "run" else cmd_day(args))


if __name__ == "__main__":
    main()
