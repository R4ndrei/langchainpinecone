#!/bin/bash

# Load environment variables from .env file
set -a  # automatically export all variables
source .env
set +a

# Run the Qdrant container
docker run -d \
  -p 6333:6333 \
  qdrant/qdrant

