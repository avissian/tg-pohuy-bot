FROM python:3.12-alpine

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .
COPY skills/ ./skills/

RUN adduser -D -u 1000 bot && mkdir -p /data && chown bot:bot /data
USER bot

ENV DATA_DIR=/data
CMD ["python", "bot.py"]
