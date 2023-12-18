import gc
import numpy
from torch import device
import torch
import utils
from rl_algorithm.drqn.agent import DRQNAgent
from rl_algorithm.dqn.agent import DQNAgent
from array2gif import write_gif
import os

# Parse arguments
import argparse

gc.collect()
torch.cuda.empty_cache()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True, help="Name of the environment to run (REQUIRED)")
parser.add_argument("--model", required=True, help="Path to the trained model (REQUIRED)")
parser.add_argument("--algorithm", type=str, choices=['dqn', 'drqn'], required=True, help="Algorithm to use: dqn or drqn (REQUIRED)")
parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
parser.add_argument("--pause", type=float, default=0.1, help="Pause duration between two consequent actions (default: 0.1)")
parser.add_argument("--gif", type=str, default=None, help="Store output as GIF with the given filename")
parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to visualize")
parser.add_argument("--shift", type=int, default=0, help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--log_wandb", type=bool, default=False)
parser.add_argument("--max-memory", type=int, default=500000, help="Maximum experiences stored (default: 500000)")
parser.add_argument("--softmax_ww", type=int, default=50)
parser.add_argument("--rnd_scale", type=float, default=None)
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)")
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

utils.seed(args.seed)

env = utils.make_env(args.env, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

model_dir = utils.get_model_dir(args.model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exploration_options = ["epsilon-random", "epsilon-z", "epsilon-rnd", "epsilon"]
obs_space, preprocess_obss = utils.get_obss_preprocessor(
    env.observation_space
)

# Load agent based on the algorithm
if args.algorithm == "dqn":
    agent = DQNAgent(
        env=env,
        eval_env=env,
        exploration_options=exploration_options,
        device=device,
        args=args,
        preprocess_obs=preprocess_obss,
        model_dir=model_dir,
        pause=args.pause,
        gif_interval=0,
    )
elif args.algorithm == "drqn":
    agent = DRQNAgent(
        env=env,
        eval_env=env,
        exploration_options=exploration_options,
        device=device,
        args=args,
        preprocess_obs=preprocess_obss,
        model_dir=model_dir,
    )
else:
    raise ValueError("Unsupported algorithm specified")

agent.load_model(args.model)
print("Agent loaded\n")

if args.gif:
    os.makedirs("gifs", exist_ok=True)

log_reward = []
# Run the agent
for episode in range(args.episodes):
    obs = env.reset()[0]
    done = False
    episode_step = 0
    frames = []

    while not done and episode_step < agent.max_episode_length:
        env.render()
        if args.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))

        preprocessed_obs = agent.preprocess_obs([obs], device=device)
        action, _ = utils.action.select_greedy_action(agent, preprocessed_obs, None)
        new_obs, reward, done, _, _ = env.step(action)
        log_reward.append(reward)
        obs = new_obs
        episode_step += 1

    print(f"Episode {episode + 1} reward: {reward}. Total Reward: {sum(log_reward)}")

    # Saving the gif if required for each episode
    if args.gif:
        gif_filename = f"gifs/{args.gif}_episode_{episode + 1}.gif"
        print(f"Saving gif for episode {episode + 1}... ", end="")
        write_gif(numpy.array(frames), gif_filename, fps=1 / args.pause)
        print("Done.")