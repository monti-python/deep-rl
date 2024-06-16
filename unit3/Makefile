train:
	python -m rl_zoo3.train --algo dqn --env SpaceInvadersNoFrameskip-v4 -f logs/ -c dqn.yml

optimize:
	python -m rl_zoo3.train --algo dqn --env SpaceInvadersNoFrameskip-v4 -f logs/ -c dqn.yml -n 50000 -optimize --n-trials 1000 --n-jobs 2 --sampler tpe --pruner median

evaluate:
	python -m rl_zoo3.enjoy  --algo dqn --env SpaceInvadersNoFrameskip-v4 --no-render --n-timesteps 5000 --folder logs/

record:
	python -m rl_zoo3.record_video --algo dqn --env SpaceInvadersNoFrameskip-v4 -f logs -n 5000

publish:
	python -m rl_zoo3.push_to_hub --algo dqn --env SpaceInvadersNoFrameskip-v4 -f logs/ -orga "monti-python" -m "Initial commit"
