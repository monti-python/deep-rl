
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
    # Download Pyramids environment
    mkdir -p ./training-envs-executables/linux
    wget "https://huggingface.co/spaces/unity/ML-Agents-Pyramids/resolve/main/Pyramids.zip" -O ./training-envs-executables/linux/Pyramids.zip
    unzip -o -d ./training-envs-executables/linux/ ./training-envs-executables/linux/Pyramids.zip
    chmod -R 755 ./training-envs-executables/linux/Pyramids/Pyramids
else
    cd ml-agents
fi

# Copy our training config for Huggy
cp ../Pyramids.yaml ./config/ppo/Pyramids_monti-python.yaml
# Train the model
mlagents-learn ./config/ppo/Pyramids_monti-python.yaml \
  --env="./training-envs-executables/linux/Pyramids/Pyramids" \
  --run-id="Pyramids1" \
  --no-graphics \
  $( [ "$RESUME" = true ] && echo "--resume" || echo "--force" )

# Upload trained model to Hub
mlagents-push-to-hf \
  --run-id="Pyramids1" \
  --local-dir=./results/Pyramids1 \
  --repo-id=monti-python/ppo-Pyramids \
  --commit-message="First training for Pyramids"
