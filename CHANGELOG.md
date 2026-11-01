# Changelog

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
