#!/bin/bash
# Health check script for Docker container

# Check if server is responding
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)

if [ "$response" = "200" ]; then
    exit 0
else
    exit 1
fi
