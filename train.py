import time
import utils
import argparse
import datetime
import torch, gc
# import numpy
import os
os.environ["KMP_DUPLICATE_LIB_OK"] ="TRUE"

from rl_algorithm.drqn.agent import DRQNAgent
from rl_algorithm.dqn.agent import DQNAgent

gc.collect()
torch.cuda.empty_cache()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True, help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None, help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=-1, help="specific seed")
parser.add_argument("--frames", type=int, default=2*10**6, help="number of frames of training (default: 2e6)")
parser.add_argument("--max-memory", type=int, default=500000, help="Maximum experiences stored (default: 500000)")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)")
parser.add_argument("--algorithm", type=str, default="dqn", help="dqn, drqn")
parser.add_argument("--rnd_scale", type=float, default=None)
parser.add_argument("--softmax_ww", type=int, default=50)
parser.add_argument("--log_wandb", type=bool, default=False)
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set run dir
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = "{}_{}_{}".format(
    args.env, args.algorithm, date
)

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()

return_per_frame, test_return_per_frame = [], []

seed = args.seed
utils.seed(seed)
env = utils.make_env(args.env, seed)
eval_env = utils.make_env(args.env, seed)

return_per_frame_, test_return_per_frame_ = [], []
num_frames = 0
episode = 0

# Load observations preprocessor
obs_space, preprocess_obss = utils.get_obss_preprocessor(
    env.observation_space
)

exploration_options = ["epsilon-random", "epsilon-z", "epsilon-rnd", "epsilon"]

if args.algorithm == "dqn":
    agent = DQNAgent(
        env=env,
        eval_env=eval_env,
        exploration_options=exploration_options,
        device=device,
        args=args,
        preprocess_obs=preprocess_obss,
        model_dir=model_dir,
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

while num_frames < args.frames:
    update_start_time = time.time()
    logs = agent.collect_experiences(
        start_time=start_time,
        episode=episode,
        num_frames=num_frames,
        return_per_frame_=return_per_frame_,
        test_return_per_frame_=test_return_per_frame_,
    )
    update_end_time = time.time()

    num_frames = logs["num_frames"]

    episode += 1

    # Save status
    if args.save_interval > 0 and episode % args.save_interval == 0:
        status = {"num_frames": num_frames, "update": episode,
                  "model_state": agent.policy_network.state_dict(),
                  "optimizer_state": agent.optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        print("Status saved")

return_per_frame.append(return_per_frame_)
test_return_per_frame.append(test_return_per_frame_)

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []



viz_env = utils.make_env(args.env, seed, render_mode="human")


# Create a window to view the environment
viz_env.render()

obs = viz_env.reset()[0]
done = False

log_loss = []
log_reward = []
episode_step = 0
while not done and episode_step < agent.max_episode_length:
    viz_env.render()
    episode_step += 1
    preprocessed_obs = agent.preprocess_obs([obs], device=agent.device)

    action, _ = utils.action.select_greedy_action(agent, preprocessed_obs, None)
    new_obs, reward, done, _, _ = viz_env.step(action)
    log_reward.append(reward)
    obs = new_obs

# if args.gif:
#     print("Saving gif... ", end="")
#     write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
#     print("Done.")
