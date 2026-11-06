import contextlib
import json
import os
import time
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TAPO_DATA") or ROOT)
SETTINGS_FILE = DATA / "panel_settings.json"
STATUS_FILE = DATA / "live_status.json"
CONTROL_FILE = DATA / "live_control.json"
CREDENTIALS_FILE = DATA / ".tapo_credentials.json"
LOCK_FILE = DATA / ".tapo_session.lock"
LOG_DIR = DATA / "logs"
LOG_LIMIT = 5 * 1024 * 1024

DEFAULTS = {"paused": False, "interval_min": 60, "retry_min": 10, "queue_gb": 2.0, "min_free_gb": 4.0,
            "memory_gate_pct": 60}
LIMITS = {"interval_min": (5, 1440), "retry_min": (1, 240), "queue_gb": (0.5, 20.0), "min_free_gb": (2.0, 50.0),
          "memory_gate_pct": (30, 90)}


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, data, mode=None):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    if mode is not None:
        with contextlib.suppress(OSError):
            os.chmod(tmp, mode)
    tmp.replace(path)


def load_settings():
    return {**DEFAULTS, **read_json(SETTINGS_FILE, {})}


def check_setting(key, value):
    low, high = LIMITS[key]
    number = float(str(value).replace(",", ".").strip())
    if not low <= number <= high:
        raise ValueError(f"{key}: {low}..{high}")
    return int(number) if isinstance(DEFAULTS[key], int) else number


def save_settings(updates):
    settings = load_settings()
    for key, value in updates.items():
        if key == "paused":
            settings[key] = bool(value)
        else:
            settings[key] = check_setting(key, value)
    write_json(SETTINGS_FILE, settings)
    return settings


def read_status():
    return read_json(STATUS_FILE, {})


def write_status(status):
    write_json(STATUS_FILE, {**status, "updated": time.time()})


def read_control():
    return read_json(CONTROL_FILE, {"wake": 0, "restart": 0})


def bump_control(key):
    control = read_control()
    control[key] = control.get(key, 0) + 1
    write_json(CONTROL_FILE, control)
    return control


def load_credentials():
    saved = read_json(CREDENTIALS_FILE, {})
    email = saved.get("email") or os.environ.get("TAPO_EMAIL", "")
    password = saved.get("password") or os.environ.get("TAPO_PASSWORD", "")
    return email, password


def save_credentials(email, password):
    write_json(CREDENTIALS_FILE, {"email": email.strip(), "password": password}, mode=0o600)


def mask_email(email):
    name, _, domain = email.partition("@")
    return f"{name[:2]}***@{domain}" if domain else "не задан"


@contextlib.contextmanager
def session_lock():
    with open(LOCK_FILE, "a", encoding="utf-8") as handle:
        if fcntl:
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl:
                fcntl.flock(handle, fcntl.LOCK_UN)


def log_path(source):
    return LOG_DIR / f"{source}.log"


def append_log(source, line):
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = log_path(source)
        if path.exists() and path.stat().st_size > LOG_LIMIT:
            path.replace(path.with_suffix(".log.1"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def tail_log(source, max_bytes=200_000):
    path = log_path(source)
    try:
        with path.open("rb") as handle:
            handle.seek(max(path.stat().st_size - max_bytes, 0))
            data = handle.read()
    except OSError:
        return ""
    return data.decode("utf-8", "replace")
