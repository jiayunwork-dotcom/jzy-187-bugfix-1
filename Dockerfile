FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIDE_DB_PATH=/data/tide.db

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Persistent data lives on the mounted volume.
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# Only the HTTP interface is exposed.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
