FROM python:3.12-slim

WORKDIR /app

# Install system deps for pipecat / silero
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Port Cloud Run will map
EXPOSE 8080
ENV PORT=8080
ENV HOST=0.0.0.0

# Cloud Run sets K_SERVICE env var — server.py uses it for wss:// detection
CMD ["python", "run.py"]
