FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
ENV TAPO_DATA=/data PYTHONUNBUFFERED=1 FFMPEG=/usr/bin/ffmpeg
RUN useradd -u 1000 -m tapo && mkdir -p /data && chown tapo /data
USER tapo
ENTRYPOINT ["python", "-u"]
CMD ["backup.py", "run"]
