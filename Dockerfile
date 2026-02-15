# Use Python 3.14 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY utils/ ./utils/
COPY training/ ./training/
COPY evaluation/ ./evaluation/
COPY configs/ ./configs/

# Copy model checkpoints and data
COPY checkpoints/ ./checkpoints/
COPY dataset/*.pkl ./dataset/
COPY dataset/user_features.csv ./dataset/
COPY dataset/item_features.csv ./dataset/

# Copy environment file
COPY .env .env

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
