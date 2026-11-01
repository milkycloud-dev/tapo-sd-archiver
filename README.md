<p align="center"><img src="assets/icon.svg" width="128" height="128" alt="Tapo SD Archiver icon"></p>

<h1 align="center">Tapo SD Archiver</h1>

<p align="center">Moves TP-Link Tapo camera recordings from the SD card through the Tapo cloud into Telegram forum topics: one chronological video feed per camera, recognized camera time, speech transcripts, sound events and content filters.</p>

<p align="center">
  <a href="https://github.com/milkycloud-dev/tapo-sd-archiver/actions/workflows/check.yml"><img src="https://github.com/milkycloud-dev/tapo-sd-archiver/actions/workflows/check.yml/badge.svg" alt="Check"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-proprietary-lightgrey" alt="License: proprietary"></a>
</p>

<p align="center"><a href="#english">English</a> | <a href="#русский">Русский</a></p>

<a id="english"></a>

## English

### Overview

Tapo cameras record to a microSD card that fills up, and the only remote access is the Tapo cloud. This project runs on a server and needs no access to the camera's network.

- **`backup.py`**: logs in to the Tapo cloud (with e-mail MFA on first run), lists SD recordings, downloads them through the cloud relay, converts them to MP4 (H.264 copy or HEVC, G.711 audio to AAC) and posts one daily compilation per camera with chapters and clickable timecodes.
- **`archive.py`**: takes an existing archive of raw SD files that was already posted to Telegram and republishes it cleanly: one topic per camera, strictly chronological, each video remuxed for instant playback, with a caption of recognized camera times.
- **Speech stream**: a second, independent stream posts speech clips with transcripts into a separate topic per camera.

### How the archive works

1. `archive.py scan` reads the source topics by message id with the download bot and records every video file.
2. For every file, in order:
   - the overlay clock is read from keyframes with digit templates calibrated automatically per resolution and font (tesseract boxes on first sight, then a fast vectorized matcher); jumps between motion events become chapters;
   - the video is remuxed with `faststart`, audio goes to AAC;
   - a sample of frames goes through an NSFW classifier; flagged videos are posted as Telegram spoilers with an `18+` mark;
   - the posting bot uploads the file in parallel parts and posts it into the camera topic.
3. The speech stream takes the same files from a bounded queue, in the same order:
   - GigaAM v3 transcription with a level and VAD gate, silence compaction and word timestamps;
   - YAMNet detection of screams, moans and crying;
   - a blacklist masks political terms, politicians' names, obscene language, 18+ terms, drugs, extremism, slurs, suicide and long digit runs (first and last character stay);
   - a clip of the speech moments is cut from the queued video and posted with the transcript.

Ordering is strict in every topic: files are prepared in parallel, but a post is sent only after every earlier post of that topic.

### Safety and resources

- No ids, tokens or addresses in the code: everything comes from `.env`.
- Two bots with fixed roles: the download bot only reads and downloads, the posting bot only posts, edits and creates topics. Tests enforce it.
- Work files are deleted right after use; the work directory is wiped on start, exit and SIGTERM; downloads pause below a free-space floor; the speech queue has a size cap.
- The container runs read-only, without capabilities, at the lowest CPU share, with a memory cap and no swap.
- Models are downloaded from pinned revisions and checked by SHA-256.

### Requirements

- Docker with Compose on the server
- A Tapo account with the cameras, `TG_API_ID` and `TG_API_HASH` from my.telegram.org, two bots that are admins of the target forum group

### Setup

```bash
cp .env.example secrets/tapo.env
docker compose build
docker compose run --rm tapo archive.py scan
docker compose --profile archive up -d archive
```

Main settings in `secrets/tapo.env`:

| Variable | Meaning |
|---|---|
| `TAPO_EMAIL`, `TAPO_PASSWORD` | Tapo cloud account |
| `TG_API_ID`, `TG_API_HASH` | Telegram API application |
| `TG_BOT_TOKEN` | posting bot |
| `TG_DL_BOT_TOKEN` | download bot |
| `TG_CHAT` | forum group id |
| `ARCHIVE_SOURCES` | `topic=camera` pairs of the source topics |
| `ARCHIVE_TOPICS` | `name=topic` pairs of the target topics, including `<camera> субтитры` |

Tuning variables (workers, transfer streams, gates, thresholds, disk floors) are read at the top of each module with their defaults.

### Commands

```bash
python archive.py scan            # index the source topics
python archive.py summary         # what will be posted
python archive.py process         # run both streams
python archive.py process --dry-run --limit 3
python archive.py reset --yes --speech-only   # delete speech posts and redo them
python backup.py run              # daily cloud backup loop
python backup.py day --camera C200 --date 20261028 --no-upload
```

### Tests

```bash
python -m unittest discover -s tests -v
```

The tests need `ffmpeg` and use mocks for the models and Telegram.

### Notes

Captions, transcripts and topic names are Russian on purpose: the cameras and their users are Russian speaking.

<a id="русский"></a>

## Русский

### Что это

Карты памяти камер TP-Link Tapo заполняются, а удалённый доступ есть только через облако Tapo. Проект работает на сервере и не требует доступа к сети камер.

- **`backup.py`**: входит в облако Tapo (при первом запуске код из почты), получает список записей с карты, скачивает их через облачный relay, переводит в MP4 (H.264 без перекодирования или HEVC, звук G.711 в AAC) и публикует одну дневную подборку на камеру с главами и кликабельными таймкодами.
- **`archive.py`**: берёт уже выложенный в Telegram архив сырых файлов с карты и публикует его заново и аккуратно: своя тема на камеру, строго по хронологии, каждое видео пересобрано для мгновенного воспроизведения, в подписи распознанное время камеры.
- **Поток речи**: второй, независимый поток выкладывает фрагменты с речью и расшифровкой в отдельную тему для каждой камеры.

### Как работает архив

1. `archive.py scan` читает исходные темы по номерам сообщений ботом-загрузчиком и запоминает все видеофайлы.
2. Для каждого файла по порядку:
   - время с оверлея читается по ключевым кадрам шаблонами цифр, которые калибруются автоматически под разрешение и шрифт (рамки tesseract при первой встрече, дальше быстрый векторный поиск); скачки между событиями движения становятся главами;
   - видео пересобирается с `faststart`, звук в AAC;
   - выборка кадров проходит классификатор NSFW; отмеченные видео публикуются спойлером Telegram с пометкой `18+`;
   - бот-публикатор заливает файл параллельными частями и публикует его в тему камеры.
3. Поток речи берёт те же файлы из ограниченной очереди в том же порядке:
   - расшифровка GigaAM v3 с отбором по уровню и VAD, вырезанием тишины и временем слов;
   - YAMNet находит крики, стоны и плач;
   - чёрный список маскирует политику, фамилии политиков, мат, 18+, наркотики, экстремизм, оскорбления, суицид и длинные числа (первый и последний символ остаются);
   - из видео в очереди вырезаются моменты с речью и публикуются с расшифровкой.

Порядок строгий в каждой теме: файлы готовятся параллельно, но пост уходит только после всех более ранних постов этой темы.

### Безопасность и ресурсы

- В коде нет номеров, токенов и адресов: всё берётся из `.env`.
- Два бота с жёсткими ролями: загрузчик только читает и скачивает, публикатор только публикует, правит и создаёт темы. Тесты это проверяют.
- Рабочие файлы удаляются сразу после использования; рабочая папка очищается при старте, выходе и SIGTERM; загрузка встаёт на паузу при малом свободном месте; очередь речи ограничена по размеру.
- Контейнер работает только для чтения, без capabilities, с минимальным приоритетом CPU, ограничением памяти и без swap.
- Модели скачиваются с закреплённых ревизий и сверяются по SHA-256.

### Что нужно

- Docker с Compose на сервере
- Аккаунт Tapo с камерами, `TG_API_ID` и `TG_API_HASH` с my.telegram.org, два бота-администратора в целевой группе с темами

### Установка

Команды и переменные окружения перечислены в английском разделе выше. Секреты лежат только в `secrets/tapo.env`.

### Тесты

```bash
python -m unittest discover -s tests -v
```

Тестам нужен `ffmpeg`, модели и Telegram подменяются заглушками.

### Примечание

Подписи, расшифровки и названия тем сделаны на русском намеренно: камеры и их пользователи русскоязычные.
