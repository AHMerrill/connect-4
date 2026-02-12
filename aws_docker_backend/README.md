# Connect 4 AWS Backend - Updated Deployment

This folder contains the corrected and updated files for deploying the Connect 4 AI backend on AWS.

## Files Included

| File | Description |
|------|-------------|
| `aws_backend.py` | Main backend server with Anvil Uplink integration |
| `model_wrappers.py` | Model loading and prediction functions |
| `connect4_engine.py` | Game logic and move validation |
| `requirements.txt` | Python dependencies |
| `.env` | Environment variables (update with your Anvil Uplink key) |
| `Dockerfile` | Docker container configuration |
| `docker-compose.yml` | Docker Compose orchestration |
| `models/cnn.h5` | CNN model file |
| `models/transformer_simple.h5` | Transformer model file |

## Key Fixes from Previous Version

1. **Real Models Used**: Backend now loads and uses actual CNN/Transformer models instead of MockModel
2. **Added `anvil.server.wait_forever()`**: Ensures the Uplink connection stays alive
3. **Single Uplink Key**: Key is only in `.env` file, not hardcoded elsewhere
4. **Fixed Model Input Shapes**: Proper board conversion for model predictions
5. **Player Format Handling**: Supports both numeric (1, 2) and string ('plus', 'minus') formats
6. **Proper Error Handling**: Graceful fallback if models fail to load

## Deployment Instructions

### 1. Update Environment Variables

Edit the `.env` file with your Anvil Uplink key:
```
ANVIL_UPLINK_KEY=your_actual_uplink_key_here
```

### 2. Transfer Files to AWS

```bash
# From your local machine
scp -r updated/* ubuntu@your-aws-ip:/home/ubuntu/connect4/
```

### 3. Build and Run with Docker

```bash
# SSH into your AWS instance
ssh ubuntu@your-aws-ip

# Navigate to the project directory
cd /home/ubuntu/connect4

# Build the Docker image
docker build -t connect4-backend:latest .

# Run with Docker Compose
docker-compose up -d
```

### 4. Verify Deployment

Check container status:
```bash
docker ps
docker logs connect4-backend
```

You should see:
```
Loading CNN model...
CNN model loaded successfully
Loading Transformer model...
Transformer model loaded successfully
Connecting to Anvil...
Connected to Anvil successfully!
```

### 5. Test Connection

In your Anvil app, the connection test should now work:
```python
result = anvil.server.call('check_connection')
# Should return: "AWS Server is Online!"
```

## Troubleshooting

### Container keeps restarting
Check logs: `docker logs connect4-backend`

### Models not loading
- Verify model files exist in `models/` directory
- Check file permissions: `chmod 644 models/*.h5`

### Anvil connection fails
- Verify uplink key in `.env` is correct
- Ensure port 443 outbound is open in AWS security group
- Check if key matches what's in your Anvil app settings

### Game not responding
- Check if backend is running: `docker ps`
- View real-time logs: `docker logs -f connect4-backend`

## Architecture

```
Anvil Frontend (Web)
        |
        | (HTTPS via Anvil Uplink)
        v
AWS Backend (Docker Container)
        |
        +-- aws_backend.py (Server)
        |
        +-- model_wrappers.py (AI Models)
        |       |
        |       +-- CNN Model (cnn.h5)
        |       +-- Transformer Model (transformer_simple.h5)
        |
        +-- connect4_engine.py (Game Logic)
```
