# Changelog

## 1.0.0

- Tapo cloud backup (`backup.py`): SD card recordings through the cloud relay, MP4 with AAC audio, daily compilation with chapters and timecodes, optional HEVC.
- Telegram archive (`archive.py`): rescan of forum topics, overlay clock recognition, faststart remux, chronological posting by camera.
- Two bots with fixed roles: one downloads, one posts.
- Speech stream: GigaAM v3 transcription, scream, moan and cry detection, speech clips in separate topics.
- Content filters: political terms and names, obscene language, 18+, drugs, extremism, slurs, suicide, long digit runs.
- 18+ video detection with Telegram spoiler posts.
- Bounded disk use, resumable state, container limits for a shared host.
