import time
import wandb
import utils
import argparse
import datetime
import torch, gc

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
parser.add_argument("--pause", type=float, default=0.2, help="Pause duration between two consequent actions of the agent (default: 0.2)")
parser.add_argument("--save-interval", type=int, default=100000, help="Interval (in frames) at which to save the model (default: 100000)")
parser.add_argument("--gif-interval", type=int, default=0, help="Interval for saving GIFs during testing (0 = never)")
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
        gif_interval=args.gif_interval,
        model_dir=model_dir,
        pause=args.pause,
    )
if args.algorithm == "drqn":
    agent = DRQNAgent(
        env=env,
        eval_env=eval_env,
        exploration_options=exploration_options,
        device=device,
        args=args,
        preprocess_obs=preprocess_obss,
        gif_interval=args.gif_interval,
        model_dir=model_dir
    )

save_interval = args.save_interval
last_save_frame = 0

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

    # Save model at specified interval
    if num_frames - last_save_frame >= save_interval or num_frames >= args.frames:
        save_path = f"{model_dir}/model_{num_frames}.pth"
        agent.save_model(save_path)
        last_save_frame = num_frames
        print(f"Model saved at frame {num_frames}")

        if args.log_wandb:
            artifact_name = f"model_at_{num_frames}_frames"
            artifact = wandb.Artifact(artifact_name, type='model')
            artifact.add_file(save_path)
            wandb.log_artifact(artifact)

if num_frames - last_save_frame > 0:
    save_path = f"{model_dir}/model_final.pth"
    agent.save_model(save_path)
    print("Final model saved")

    if args.log_wandb:
        artifact = wandb.Artifact('final_model', type='model')
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)

return_per_frame.append(return_per_frame_)
test_return_per_frame.append(test_return_per_frame_)