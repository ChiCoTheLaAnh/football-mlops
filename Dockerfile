FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models

EXPOSE 8000

# Render/Railway provide the PORT env var dynamically in Docker environments.
# Fallback to 8000 for local runs.
ENV PORT=8000

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
