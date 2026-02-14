# Connect 4 AWS Backend

This folder contains the Docker-based backend for the Connect 4 AI game, hosted on AWS Lightsail.

## Files Included

| File | Description |
|------|-------------|
| `aws_backend.py` | Main backend server with Anvil Uplink integration |
| `model_wrappers.py` | Model loading and prediction functions |
| `connect4_engine.py` | Game logic and move validation |
| `requirements.txt` | Python dependencies |
| `.env.example` | Example environment variables (copy to `.env`) |
| `Dockerfile` | Docker container configuration |
| `docker-compose.yml` | Docker Compose orchestration |
| `start.sh` | Startup script for non-Docker use |
| `models/` | Placeholder for model files (see below) |

## Model Files (Not Included)

The trained model files are too large for Git (~150MB total). Download them separately:

| File | Size | Location |
|------|------|----------|
| `cnn.h5` | ~142MB | Place in `models/` folder |
| `transformer.weights.h5` | ~10MB | Place in `models/` folder |

**Models are available in the GitHub Releases or can be trained using the notebooks in `/network_training/`.**

## Deployment Instructions

### 1. Setup Environment Variables

```bash
cp .env.example .env
# Edit .env with your Anvil Uplink key
```

### 2. Transfer Files to AWS Lightsail

Using FileZilla or SCP:
```bash
scp -r aws_docker_backend/* bitnami@your-aws-ip:/home/bitnami/connect4-backend/
```

### 3. Add Model Files

Transfer your trained models to the AWS instance:
```bash
scp models/cnn.h5 bitnami@your-aws-ip:/home/bitnami/connect4-backend/models/
scp models/transformer.weights.h5 bitnami@your-aws-ip:/home/bitnami/connect4-backend/models/
```

### 4. Build and Run with Docker

```bash
# SSH into your AWS instance
ssh bitnami@your-aws-ip

# Navigate to the project directory
cd /home/bitnami/connect4-backend

# Build the Docker image (use --network=host if DNS issues)
sudo docker build --network=host -t my-anvil-app .

# Run with Docker Compose
sudo docker compose up -d

# Check logs
sudo docker compose logs
```

### 5. Verify Deployment

You should see:
```
✅ CNN model loaded successfully
✅ Transformer model loaded successfully
✅ Connected to Anvil successfully!
Backend ready - waiting for game requests...
```

## Troubleshooting

### Container keeps restarting
```bash
sudo docker compose logs
```

### Models not loading
- Verify model files exist: `ls -la models/`
- Check volume mapping in `docker-compose.yml`

### Anvil connection fails
- Verify uplink key in `.env` or `docker-compose.yml`
- Ensure outbound HTTPS (port 443) is open

### Rebuild after changes
```bash
sudo docker compose down
sudo docker build --network=host -t my-anvil-app .
sudo docker compose up -d
```

## Architecture

```
Anvil Frontend (Web)
        |
        | (HTTPS via Anvil Uplink)
        v
AWS Lightsail (Docker Container)
        |
        +-- aws_backend.py (Server)
        |
        +-- model_wrappers.py (AI Models)
        |       |
        |       +-- CNN Model (cnn.h5)
        |       +-- Transformer Model (transformer.weights.h5)
        |
        +-- connect4_engine.py (Game Logic)
```

## Team

- Alina Hota
- Arturo Juarez
- Zan Merrill
- Rohini Sondole

**Course:** MSBA Optimization II - Spring 2026, UT Austin
