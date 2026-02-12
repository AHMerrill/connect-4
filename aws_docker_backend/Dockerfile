# Connect 4 Backend Dockerfile - UPDATED VERSION
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY aws_backend.py .
COPY model_wrappers.py .
COPY connect4_engine.py .

# Create models directory (models will be mounted via docker-compose)
RUN mkdir -p /app/models /app/logs

# Environment variable will be set via docker-compose or .env
# Do NOT hardcode the uplink key here

# Expose port (for health checks if needed)
EXPOSE 3030

# Run backend
CMD ["python", "aws_backend.py"]
