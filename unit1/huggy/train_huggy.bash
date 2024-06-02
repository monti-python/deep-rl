# Clone ml-agents toolkit
git clone --depth 1 https://github.com/Unity-Technologies/ml-agents
git checkout a66ffbf0628e712758eae78d694c09930f1e4545
# Install ml-agents
cd ml-agents
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents
# Download Huggy environment
mkdir -p ./trained-envs-executables/linux
wget "https://github.com/huggingface/Huggy/raw/main/Huggy.zip" -O ./trained-envs-executables/linux/Huggy.zip
unzip -d ./trained-envs-executables/linux/ ./trained-envs-executables/linux/Huggy.zip
chmod -R 755 ./trained-envs-executables/linux/Huggy
# Copy our training config for Huggy
cp ../Huggy.yaml ./config/ppo/Huggy.yaml
# Train the model (use the same command with the `--resume` flag in case of failure)
mlagents-learn ./config/ppo/Huggy.yaml \
  --env="./trained-envs-executables/linux/Huggy/Huggy" \
  --run-id="Huggy" \
  --no-graphics
# Upload trained model to Hub
mlagents-push-to-hf \
  --run-id="Huggy" \
  --local-dir=./results/Huggy \
  --repo-id=monti-python/ppo-Huggy \
  --commit-message="Push Huggy to the Hub"
