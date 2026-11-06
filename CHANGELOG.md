# Changelog

## 1.2.0

- Live pull (`live.py`): one compilation per completed camera day from the Tapo cloud into the archive topics, split above 2 GB, strict day order, independent speech stream, resume after a restart without repeating posted parts.
- Discord panel (`panel.py`): login with the e-mail code, session checks and daily refresh, camera and SD card status, pause, restart, settings, logs and a journal.
- The live pull waits for a valid session instead of attempting a login by itself.
- Peak memory protection: memory mapped keyframe crops, one OCR worker, a memory gate before heavy stages, large buffers returned to the OS.
- Wider overlay crop: the largest font no longer crashes files, and layouts that do not fit the crop are rejected.

## 1.1.0

- Overlay recognition: tesseract upscale follows the crop height with one larger retry, a binarized pass, trailing scene text ignored; files whose overlay font changes midway are read with several layouts. 1280x720 files are now read.
- Faster calibration: at most about 300 sampled frames.
- Lower peak memory: keyframe crops go through a file instead of a pipe buffer, smaller matcher batches.
- Removed the 18+ video classifier, spoiler posts and the `18+` mark: it flagged ordinary videos.
- `reset --from NAME` redoes one camera's posts (or with `--speech-only` its speech posts) from a file on, keeping the order.
- Overflow-free sigmoid in sound event detection.

## 1.0.0

- Tapo cloud backup (`backup.py`): SD card recordings through the cloud relay, MP4 with AAC audio, daily compilation with chapters and timecodes, optional HEVC.
- Telegram archive (`archive.py`): rescan of forum topics, overlay clock recognition, faststart remux, chronological posting by camera.
- Two bots with fixed roles: one downloads, one posts.
- Speech stream: GigaAM v3 transcription, scream, moan and cry detection, speech clips in separate topics.
- Content filters: political terms and names, obscene language, 18+, drugs, extremism, slurs, suicide, long digit runs.
- 18+ video detection with Telegram spoiler posts.
- Bounded disk use, resumable state, container limits for a shared host.
