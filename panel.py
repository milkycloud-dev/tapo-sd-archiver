import asyncio
import io
import logging
import os
import re
import shutil
import time
from datetime import datetime

import discord
from ontapo import OnTapo, OnTapoError
from ontapo.errors import AuthError, MFARequired

import backup
import control
from backup import log

TOKEN = os.environ.get("PANEL_DISCORD_TOKEN", "")
GUILD_ID = int(os.environ.get("PANEL_GUILD_ID", "0") or 0)
ADMIN_IDS = {int(x) for x in os.environ.get("PANEL_ADMIN_IDS", "").replace(" ", "").split(",") if x}
CATEGORY = os.environ.get("PANEL_CATEGORY", "📷 Tapo: панель")
ARCHIVE_STATE = control.DATA / "archive_state.json"
PANEL_STATE = control.DATA / "panel_state.json"
SPEECH_DIR = control.DATA / "speech_queue"
HEARTBEAT = control.DATA / "panel_heartbeat"
REFRESH_SEC = 60
FEED_SEC = 20
PROBE_SEC = 1800
KEEPALIVE_SEC = 86400
CODE_TTL = 600
RESEND_SEC = 60
ARCHIVE_ACTIVE_SEC = 900
STALE_SEC = 180
LOG_SOURCES = {"live": "Живой поток", "archive": "Архив", "panel": "Панель"}
FEED_RE = re.compile(r"posted$|failed|gave up|login|restart|panel|FAILED|no clips|work dir cleared")
STATES = {"running": "работает", "sleeping": "ждёт новый день", "retrying": "ошибка, повторит позже",
          "waiting": "ждёт место или очередь субтитров", "paused": "на паузе", "login_required": "нужен вход",
          "starting": "запускается"}
GREEN, YELLOW, RED = 0x28BA57, 0xE0A800, 0xBB292B


class LoginFlow:
    def __init__(self):
        self.pending = None
        self.sent_at = 0.0

    @property
    def waiting(self):
        return self.pending is not None and time.monotonic() - self.sent_at < CODE_TTL

    async def drop(self):
        pending, self.pending = self.pending, None
        if pending is not None:
            await pending.aclose()

    async def finish(self, session):
        with control.session_lock():
            backup.save_session(session)
        await session.aclose()
        log("login: new session saved")

    async def start(self):
        if self.waiting and time.monotonic() - self.sent_at < RESEND_SEC:
            return "code_sent"
        await self.drop()
        email, password = control.load_credentials()
        if not email or not password:
            raise AuthError("no credentials")
        try:
            session = await OnTapo.login(email, password)
        except MFARequired as exc:
            self.pending = exc.session
            try:
                await self.pending.send_mfa_code()
            except OnTapoError:
                await self.drop()
                raise
            self.sent_at = time.monotonic()
            log("login: e-mail code requested")
            return "code_sent"
        await self.finish(session)
        return "ok"

    async def submit(self, code):
        if not self.waiting:
            await self.drop()
            raise TimeoutError
        await self.pending.submit_mfa(code.strip())
        session, self.pending = self.pending, None
        await self.finish(session)


async def probe_session():
    result = {"checked": time.time(), "cameras": []}
    try:
        s = await backup.open_saved_session()
    except backup.LoginRequired as exc:
        return {**result, "valid": False, "error": str(exc)}
    except Exception as exc:
        return {**result, "valid": None, "error": type(exc).__name__}
    async with s:
        cams = await s.cameras()
        for cam in cams:
            s.app_server = cam.app_server or s.app_server
            info = {"name": cam.name}
            try:
                info["sd"] = (await s.device(cam.device_id, name=cam.name).sd_card_status()).summary[:120]
            except Exception as exc:
                info["sd"] = f"нет ответа: {type(exc).__name__}"
            result["cameras"].append(info)
    return {**result, "valid": True}


async def keep_alive():
    saved = control.read_json(backup.SESSION_FILE, None)
    if not saved or not saved.get("refreshToken"):
        return False
    s = OnTapo.from_dict(saved)
    try:
        with control.session_lock():
            current = control.read_json(backup.SESSION_FILE, saved)
            if current.get("refreshToken") != s.refresh_token:
                return True
            await s.refresh()
            backup.save_session(s)
        log("login: session refreshed")
        return True
    except OnTapoError as exc:
        log(f"login: refresh failed: {type(exc).__name__}")
        return False
    finally:
        await s.aclose()


def archive_summary():
    state = control.read_json(ARCHIVE_STATE, {})
    items = state.get("items", {})
    cameras = {}
    for entry in items.values():
        if entry.get("messages"):
            cam = cameras.setdefault(entry.get("camera", "?"), {"posts": 0, "speech": 0, "pending": 0, "last": ""})
            cam["posts"] += 1
            cam["speech"] += 1 if entry.get("speech") else 0
            cam["pending"] += 0 if "speech" in entry else 1
            if entry.get("live") and entry["name"] > cam["last"]:
                cam["last"] = entry["name"]
    return cameras


def ago(ts):
    seconds = int(time.time() - ts)
    if seconds < 90:
        return f"{seconds} с назад"
    if seconds < 5400:
        return f"{seconds // 60} мин назад"
    return f"{seconds // 3600} ч назад"


def day_label(date):
    return f"{date[6:8]}.{date[4:6]}.{date[:4]}" if len(date) >= 8 else date


def snapshot():
    usage = shutil.disk_usage(control.DATA)
    queue = sum(p.stat().st_size for p in SPEECH_DIR.glob("*") if p.is_file()) if SPEECH_DIR.exists() else 0
    archive_age = time.time() - ARCHIVE_STATE.stat().st_mtime if ARCHIVE_STATE.exists() else 1e9
    return {"status": control.read_status(), "settings": control.load_settings(), "archive": archive_summary(),
            "free_gb": usage.free / 1e9, "queue_gb": queue / 1e9, "now": time.time(), "archive_age": archive_age}


def render(snap, probe, waiting_code):
    status, settings = snap["status"], snap["settings"]
    age = snap["now"] - status.get("updated", 0)
    alive = age < STALE_SEC
    login_needed = probe.get("valid") is False or (alive and status.get("login") == "required")
    state = STATES.get(status.get("state"), status.get("state", "?")) if alive else "не запущен"
    if settings["paused"] and alive:
        state = "на паузе"
    color = RED if login_needed else YELLOW if not alive or status.get("last_error") or settings["paused"] else GREEN
    embed = discord.Embed(title="Tapo: панель", color=color)
    if waiting_code:
        login = "⏳ Ждём код с почты: нажмите «Ввести код»"
    elif login_needed:
        login = "🔴 Нужен вход: нажмите «Войти»"
    elif probe.get("valid"):
        login = f"🟢 Действует, проверено {ago(probe['checked'])}"
    elif probe:
        login = f"🟡 Не удалось проверить: {probe.get('error', '?')}"
    else:
        login = "⚪ Ещё не проверялся"
    email, _ = control.load_credentials()
    embed.add_field(name="Вход Tapo", value=f"{login}\nАккаунт: {control.mask_email(email)}", inline=False)
    process = f"{'🟢' if alive else '⚫'} {state}"
    if not alive and snap.get("archive_age", 1e9) < ARCHIVE_ACTIVE_SEC:
        process += "\nСейчас работает архив, живой поток запустится после него"
    if alive and status.get("next_check") and status.get("state") in ("sleeping", "retrying"):
        process += f"\nСледующая проверка <t:{int(status['next_check'])}:R>"
    if alive and status.get("last_error"):
        process += f"\n⚠️ {status['last_error'][:200]}"
    if alive and status.get("speech_pending"):
        process += f"\nСубтитры: в очереди {status['speech_pending']}"
    embed.add_field(name="Живой поток", value=process, inline=False)
    sd_cards = {c["name"]: c.get("sd", "") for c in probe.get("cameras", [])}
    names = sorted(set(snap["archive"]) | set(status.get("cameras", {})) | set(sd_cards))
    for name in names:
        cam, live, lines = snap["archive"].get(name, {}), status.get("cameras", {}).get(name, {}) if alive else {}, []
        lines.append(f"Постов {cam.get('posts', 0)}, с субтитрами {cam.get('speech', 0)}")
        if cam.get("pending"):
            lines.append(f"Субтитры ждут: {cam['pending']}")
        if cam.get("last"):
            lines.append(f"Последний день: {day_label(cam['last'])}")
        if live.get("date"):
            lines.append(f"Сейчас: {day_label(live['date'])}, клипы {live.get('clips') or '...'}")
        if live.get("pending"):
            lines.append(f"Дней осталось: {live['pending']}")
        if live.get("error"):
            lines.append(f"⚠️ {live['error'][:120]}")
        if sd_cards.get(name):
            lines.append(f"Карта: {sd_cards[name]}")
        embed.add_field(name=name, value="\n".join(lines), inline=True)
    embed.add_field(name="Диск и настройки", inline=False, value=(
        f"Свободно {snap['free_gb']:.1f} ГБ, очередь {snap['queue_gb']:.1f} ГБ\n"
        f"Проверка каждые {settings['interval_min']} мин, повтор через {settings['retry_min']} мин, "
        f"очередь до {settings['queue_gb']} ГБ, диск от {settings['min_free_gb']} ГБ, "
        f"память до {settings['memory_gate_pct']}%"))
    embed.set_footer(text=f"обновлено {datetime.now():%d.%m %H:%M:%S}")
    return embed, login_needed


class CredentialsModal(discord.ui.Modal, title="Логин и пароль Tapo"):
    email = discord.ui.TextInput(label="Почта", max_length=200)
    password = discord.ui.TextInput(label="Пароль", max_length=200)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.email.default = control.load_credentials()[0]

    async def on_submit(self, interaction):
        control.save_credentials(str(self.email.value), str(self.password.value))
        await self.app.audit(interaction.user, "изменил логин и пароль")
        await self.app.run_login(interaction)


class CodeModal(discord.ui.Modal, title="Код с почты"):
    code = discord.ui.TextInput(label="Код из письма TP-Link", min_length=4, max_length=12)

    def __init__(self, app):
        super().__init__()
        self.app = app

    async def on_submit(self, interaction):
        await interaction.response.defer(ephemeral=True)
        try:
            await self.app.flow.submit(str(self.code.value))
        except TimeoutError:
            await interaction.followup.send("Код устарел. Нажмите «Войти», придёт новый.", ephemeral=True)
            return
        except Exception as exc:
            await self.app.audit(interaction.user, f"код не принят: {type(exc).__name__}")
            await interaction.followup.send(f"Код не принят: {type(exc).__name__}. Попробуйте ещё раз.", ephemeral=True)
            return
        await self.app.audit(interaction.user, "вошёл в Tapo")
        await self.app.after_login()
        await interaction.followup.send("✅ Вход выполнен, сессия сохранена.", ephemeral=True)


class SettingsModal(discord.ui.Modal, title="Настройки"):
    FIELDS = (("interval_min", "Проверка новых дней, мин (5-1440)"), ("retry_min", "Повтор после ошибки, мин (1-240)"),
              ("queue_gb", "Очередь субтитров, ГБ (0.5-20)"), ("min_free_gb", "Минимум диска, ГБ (2-50)"),
              ("memory_gate_pct", "Порог памяти, % (30-90)"))

    def __init__(self, app):
        super().__init__()
        self.app = app
        settings = control.load_settings()
        self.inputs = {}
        for key, label in self.FIELDS:
            field = discord.ui.TextInput(label=label, default=str(settings[key]), max_length=8)
            self.inputs[key] = field
            self.add_item(field)

    async def on_submit(self, interaction):
        updates = {key: str(field.value) for key, field in self.inputs.items()}
        try:
            control.save_settings(updates)
        except ValueError as exc:
            await interaction.response.send_message(f"Не сохранено, значение вне диапазона: {exc}", ephemeral=True)
            return
        control.bump_control("wake")
        await self.app.audit(interaction.user, "изменил настройки: " + ", ".join(f"{k}={v}" for k, v in updates.items()))
        await interaction.response.send_message("✅ Сохранено, живой поток применит сразу.", ephemeral=True)
        await self.app.refresh()


class LogSelect(discord.ui.Select):
    def __init__(self):
        super().__init__(placeholder="Какой журнал показать",
                         options=[discord.SelectOption(label=label, value=key) for key, label in LOG_SOURCES.items()])

    async def callback(self, interaction):
        source = self.values[0]
        text = control.tail_log(source)
        if not text:
            await interaction.response.send_message("Журнал пока пуст.", ephemeral=True)
            return
        last = "\n".join(text.splitlines()[-22:])[-1800:]
        await interaction.response.send_message(
            f"```\n{last}\n```", ephemeral=True,
            file=discord.File(io.BytesIO(text.encode("utf-8")), filename=f"{source}.log"))


class LogView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=300)
        self.add_item(LogSelect())


class ConfirmRestart(discord.ui.View):
    def __init__(self, app):
        super().__init__(timeout=60)
        self.app = app

    @discord.ui.button(label="Да, перезапустить", style=discord.ButtonStyle.danger)
    async def yes(self, interaction, _):
        control.bump_control("restart")
        await self.app.audit(interaction.user, "перезапустил живой поток")
        await interaction.response.edit_message(content="♻️ Перезапуск, незавершённый день продолжится.", view=None)


class PanelView(discord.ui.View):
    def __init__(self, app, paused=False):
        super().__init__(timeout=None)
        self.app = app
        self.pause.label = "Продолжить" if paused else "Пауза"
        self.pause.emoji = "▶️" if paused else "⏸️"

    async def interaction_check(self, interaction):
        if self.app.allowed(interaction.user):
            return True
        await self.app.audit(interaction.user, "отказано: " + str(interaction.data.get("custom_id")))
        await interaction.response.send_message("Только для владельца.", ephemeral=True)
        return False

    @discord.ui.button(label="Войти", emoji="🔑", style=discord.ButtonStyle.primary, custom_id="tp:login", row=0)
    async def login(self, interaction, _):
        await self.app.audit(interaction.user, "нажал «Войти»")
        await self.app.run_login(interaction)

    @discord.ui.button(label="Ввести код", emoji="🔢", style=discord.ButtonStyle.success, custom_id="tp:code", row=0)
    async def code(self, interaction, _):
        if not self.app.flow.waiting:
            await interaction.response.send_message("Код сейчас не ждём. Сначала нажмите «Войти».", ephemeral=True)
            return
        await interaction.response.send_modal(CodeModal(self.app))

    @discord.ui.button(label="Логин и пароль", emoji="✏️", style=discord.ButtonStyle.secondary, custom_id="tp:creds", row=0)
    async def creds(self, interaction, _):
        await interaction.response.send_modal(CredentialsModal(self.app))

    @discord.ui.button(label="Проверить сейчас", emoji="🔄", style=discord.ButtonStyle.secondary, custom_id="tp:check", row=1)
    async def check(self, interaction, _):
        await interaction.response.defer(ephemeral=True)
        control.bump_control("wake")
        self.app.probe = await probe_session()
        await self.app.refresh()
        await interaction.followup.send("Проверено, живой поток разбужен.", ephemeral=True)

    @discord.ui.button(label="Пауза", emoji="⏸️", style=discord.ButtonStyle.secondary, custom_id="tp:pause", row=1)
    async def pause(self, interaction, _):
        paused = not control.load_settings()["paused"]
        control.save_settings({"paused": paused})
        await self.app.audit(interaction.user, "поставил на паузу" if paused else "снял с паузы")
        note = "Пауза начнётся перед следующим днём." if paused else "Продолжаем."
        await interaction.response.send_message(note, ephemeral=True)
        await self.app.refresh()

    @discord.ui.button(label="Перезапуск", emoji="♻️", style=discord.ButtonStyle.secondary, custom_id="tp:restart", row=1)
    async def restart(self, interaction, _):
        await interaction.response.send_message("Перезапустить живой поток?", view=ConfirmRestart(self.app), ephemeral=True)

    @discord.ui.button(label="Настройки", emoji="⚙️", style=discord.ButtonStyle.secondary, custom_id="tp:settings", row=2)
    async def settings(self, interaction, _):
        await interaction.response.send_modal(SettingsModal(self.app))

    @discord.ui.button(label="Журналы", emoji="📜", style=discord.ButtonStyle.secondary, custom_id="tp:logs", row=2)
    async def logs(self, interaction, _):
        await interaction.response.send_message("Выберите журнал:", view=LogView(), ephemeral=True)


class Panel(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents(guilds=True))
        self.flow = LoginFlow()
        self.probe = {}
        self.state = control.read_json(PANEL_STATE, {})
        self.panel_channel = None
        self.journal = None
        self.alerted = False
        self.started = False

    def allowed(self, user):
        guild = self.get_guild(GUILD_ID)
        return user.id in ADMIN_IDS or (guild is not None and user.id == guild.owner_id)

    def save_state(self):
        control.write_json(PANEL_STATE, self.state)

    async def setup_hook(self):
        self.add_view(PanelView(self))

    async def on_ready(self):
        if self.started:
            return
        self.started = True
        await self.ensure_channels()
        for job in (self.refresh_loop, self.feed_loop, self.probe_loop, self.keepalive_loop):
            asyncio.create_task(self.guarded(job))
        log("panel started")

    async def guarded(self, job):
        while True:
            try:
                await job()
            except Exception as exc:
                log(f"panel {job.__name__} failed: {type(exc).__name__}: {exc}")
                await asyncio.sleep(30)

    async def ensure_channels(self):
        guild = self.get_guild(GUILD_ID) or await self.fetch_guild(GUILD_ID)
        overwrites = {guild.default_role: discord.PermissionOverwrite(view_channel=False),
                      guild.me: discord.PermissionOverwrite(view_channel=True, send_messages=True, embed_links=True,
                                                            attach_files=True, read_message_history=True)}
        for admin in ADMIN_IDS:
            overwrites[discord.Object(id=admin)] = discord.PermissionOverwrite(view_channel=True, send_messages=False)
        category = guild.get_channel(self.state.get("category_id", 0)) or discord.utils.get(guild.categories, name=CATEGORY)
        if category is None:
            category = await guild.create_category(CATEGORY, overwrites=overwrites, reason="tapo-panel")
        self.state["category_id"] = category.id
        for attr, name, key in (("panel_channel", "панель", "panel_id"), ("journal", "журнал", "journal_id")):
            channel = guild.get_channel(self.state.get(key, 0)) or discord.utils.get(category.text_channels, name=name)
            if channel is None:
                channel = await guild.create_text_channel(name, category=category, overwrites=overwrites, reason="tapo-panel")
            self.state[key] = channel.id
            setattr(self, attr, channel)
        self.save_state()

    async def audit(self, user, action):
        log(f"panel: {user} {action}")
        if self.journal:
            await self.journal.send(f"👤 {user.mention} {action}", allowed_mentions=discord.AllowedMentions.none())

    async def run_login(self, interaction):
        await interaction.response.defer(ephemeral=True)
        try:
            result = await self.flow.start()
        except AuthError as exc:
            log(f"login: rejected: {type(exc).__name__}")
            await interaction.followup.send("❌ Tapo отклонил вход. Проверьте логин и пароль кнопкой «Логин и пароль».",
                                            ephemeral=True)
            return
        except Exception as exc:
            log(f"login: failed: {type(exc).__name__}")
            await interaction.followup.send(f"❌ Не вышло: {type(exc).__name__}. Попробуйте позже.", ephemeral=True)
            return
        if result == "code_sent":
            email = control.mask_email(control.load_credentials()[0])
            await interaction.followup.send(f"📧 Код отправлен на {email}. Нажмите «Ввести код», код живёт 10 минут.",
                                            ephemeral=True)
        else:
            await self.after_login()
            await interaction.followup.send("✅ Вход выполнен без кода, сессия сохранена.", ephemeral=True)
        await self.refresh()

    async def after_login(self):
        self.alerted = False
        self.probe = await probe_session()
        control.bump_control("wake")
        await self.refresh()

    async def refresh(self):
        snap = await asyncio.to_thread(snapshot)
        embed, login_needed = render(snap, self.probe, self.flow.waiting)
        view = PanelView(self, snap["settings"]["paused"])
        message = None
        if self.state.get("panel_message"):
            try:
                message = await self.panel_channel.fetch_message(self.state["panel_message"])
            except discord.NotFound:
                message = None
        if message:
            await message.edit(embed=embed, view=view)
        else:
            message = await self.panel_channel.send(embed=embed, view=view)
            self.state["panel_message"] = message.id
            self.save_state()
        HEARTBEAT.touch()
        if login_needed and not self.alerted:
            self.alerted = True
            guild = self.get_guild(GUILD_ID)
            owner = f"<@{guild.owner_id}> " if guild else ""
            await self.journal.send(f"🔴 {owner}Tapo просит вход, живой поток ждёт. Нажмите «Войти» в {self.panel_channel.mention}.")
        elif not login_needed:
            self.alerted = False

    async def refresh_loop(self):
        while True:
            await self.refresh()
            await asyncio.sleep(REFRESH_SEC)

    async def probe_loop(self):
        while True:
            self.probe = await probe_session()
            await asyncio.sleep(PROBE_SEC)

    async def keepalive_loop(self):
        while True:
            if time.time() - self.state.get("keepalive", 0) >= KEEPALIVE_SEC:
                if await keep_alive():
                    self.state["keepalive"] = time.time()
                    self.save_state()
            await asyncio.sleep(3600)

    async def feed_loop(self):
        offsets = self.state.setdefault("feed", {})
        while True:
            lines = []
            for source in ("live", "archive"):
                path = control.log_path(source)
                if not path.exists():
                    continue
                size = path.stat().st_size
                offset = offsets.get(source)
                if offset is None or offset > size:
                    offsets[source] = size
                    continue
                if offset < size:
                    with path.open("rb") as handle:
                        handle.seek(offset)
                        data = handle.read(200_000)
                    chunk = data[:data.rfind(b"\n") + 1]
                    offsets[source] = offset + len(chunk)
                    lines += [line for line in chunk.decode("utf-8", "replace").splitlines() if FEED_RE.search(line)]
            if lines:
                self.save_state()
                text = "\n".join(lines)
                for start in range(0, len(text), 1900):
                    await self.journal.send(f"```\n{text[start:start + 1900]}\n```")
            await asyncio.sleep(FEED_SEC)


def main():
    logging.basicConfig(level=logging.WARNING)
    if not TOKEN or not GUILD_ID:
        raise SystemExit("PANEL_DISCORD_TOKEN and PANEL_GUILD_ID must be set")
    Panel().run(TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
