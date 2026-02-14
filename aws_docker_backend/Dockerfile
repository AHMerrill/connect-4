# Lightweight Docker image for Connect 4 backend
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies (use --no-cache-dir to save space)
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for models
RUN mkdir -p /app_data
RUN chmod -R 777 /app_data

# Copy application code
COPY aws_backend.py .
COPY model_wrappers.py .
COPY connect4_engine.py .

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Run the backend
CMD ["python", "aws_backend.py"]
