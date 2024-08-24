
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
    # Download SoccerTwos environment
    mkdir -p ./training-envs-executables/linux
    wget "https://drive.usercontent.google.com/download?id=1KuqBKYiXiIcU4kNMqEzhgypuFP5_45CL&export=download&authuser=0&confirm=t&uuid=2699f944-9ff8-4b03-b5b2-40ebd4e18b16&at=AO7h07fFQV4WhI-FDwVNDbt-DUH4%3A1724365370196" -O ./training-envs-executables/linux/SoccerTwos.zip
    unzip -o -d ./training-envs-executables/linux/SoccerTwos ./training-envs-executables/linux/SoccerTwos.zip
    chmod -R 755 ./training-envs-executables/linux/SoccerTwos
else
    cd ml-agents
fi

# Copy our training config
cp ../SoccerTwos.yaml ./config/poca/SoccerTwos_monti-python.yaml
# Train the model
mlagents-learn ./config/poca/SoccerTwos_monti-python.yaml \
  --env="./training-envs-executables/linux/SoccerTwos/SoccerTwos.x86_64" \
  --run-id="SoccerTwos1" \
  --no-graphics \
  $( [ "$RESUME" = true ] && echo "--resume" || echo "--force" )

# Upload trained model to Hub
mlagents-push-to-hf \
  --run-id="SoccerTwos1" \
  --local-dir=./results/SoccerTwos1 \
  --repo-id=monti-python/poca-SoccerTwos \
  --commit-message="First training for SoccerTwos"
