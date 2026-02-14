#!/bin/bash
# Start the Connect 4 backend server
# Run with: ./start.sh

source .env 2>/dev/null || true
python aws_backend.py
