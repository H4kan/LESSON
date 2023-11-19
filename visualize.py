import argparse
import numpy
import time

import utils
from utils import device
import os
os.environ["KMP_DUPLICATE_LIB_OK"] ="TRUE"

from rl_algorithm.drqn.agent import DRQNAgent
from rl_algorithm.dqn.agent import DQNAgent
# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--algorithm", type=str, default="dqn", help="dqn, drqn")
parser.add_argument("--max-memory", type=int, default=500000, help="Maximum experiences stored (default: 500000)")
parser.add_argument("--softmax_ww", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)")
parser.add_argument("--rnd_scale", type=float, default=None)

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed, render_mode="human")
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

obs_space, preprocess_obss = utils.get_obss_preprocessor(
    env.observation_space
)


exploration_options = ["epsilon-random", "epsilon-z", "epsilon-rnd", "epsilon"]

# Load agent

model_dir = utils.get_model_dir(args.model)
if args.algorithm == "dqn":
    agent = DQNAgent(
        env=env,
        eval_env=env,
        exploration_options=exploration_options,
        device=device,
        args=args,
        preprocess_obs=preprocess_obss,
        model_dir=model_dir,
        preload=False
    )
if args.algorithm == "drqn":
    agent = DRQNAgent(
        env=env,
        eval_env=eval_env,
        exploration_options=exploration_options,
        device=device,
        args=args,
        preprocess_obs=preprocess_obss,
        model_dir=model_dir,
    )

print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
obs = env.reset()[0]

done = False

log_loss = []
log_reward = []
episode_step = 0
while not done and episode_step < agent.max_episode_length:
    # time.sleep(1)
    env.render()
    if args.gif:
            frames.append(numpy.moveaxis(env.get_frame(), 2, 0))
    episode_step += 1
    preprocessed_obs = agent.preprocess_obs([obs], device=agent.device)

    action, _ = utils.action.select_greedy_action(agent, preprocessed_obs, None)
    print(action)
    new_obs, reward, done, _, _ = env.step(action)
    log_reward.append(reward)
    obs = new_obs

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
