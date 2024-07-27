#!/bin/bash

set -xeuo pipefail

RESUME=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --resume) RESUME=true ;;
    esac
    shift
done

if [ "$RESUME" = false ]; then
    # Fetch ml-agents toolkit
    git init ml-agents
    cd ml-agents
    URL=https://github.com/Unity-Technologies/ml-agents
    git remote add origin $URL || git remote set-url origin $URL
    git fetch --depth 1 origin main
    git checkout main
    # Install ml-agents
    pip3 install -e ./ml-agents-envs
    pip3 install -e ./ml-agents
    # Download SnowballTarget environment
    mkdir -p ./training-envs-executables/linux
    wget "https://github.com/huggingface/Snowball-Target/raw/main/SnowballTarget.zip" -O ./training-envs-executables/linux/SnowballTarget.zip
    unzip -o -d ./training-envs-executables/linux/ ./training-envs-executables/linux/SnowballTarget.zip
    chmod -R 755 ./training-envs-executables/linux/SnowballTarget
else
    cd ml-agents
fi

# Copy our training config for Huggy
cp ../SnowballTarget.yaml ./config/ppo/SnowballTarget.yaml
# Train the model
mlagents-learn ./config/ppo/SnowballTarget.yaml \
  --env="./training-envs-executables/linux/SnowballTarget/SnowballTarget.x86_64" \
  --run-id="SnowballTarget1" \
  --no-graphics \
  $( [ "$RESUME" = true ] && echo "--resume" || echo "--force" )

# Upload trained model to Hub
mlagents-push-to-hf \
  --run-id="SnowballTarget1" \
  --local-dir=./results/SnowballTarget1 \
  --repo-id=monti-python/ppo-SnowballTarget \
  --commit-message="First training for SnowballTarget"
