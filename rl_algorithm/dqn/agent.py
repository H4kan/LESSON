import datetime
import os
import random

import torch
import utils
import wandb
import numpy as np
from torch.distributions import Bernoulli
# from array2gif import write_gif

from rl_algorithm.dqn.replay_memory import ReplayMemory
from rl_algorithm.dqn.rnd import RND
from rl_algorithm.dqn.model import DQN
from rl_algorithm.dqn.config import batch_size
from rl_algorithm.common.option_model import OptionQ

class DQNAgent:
    """
    The Deep Q Learning algorithm
    """

    def __init__(
        self,
        env,
        eval_env,
        exploration_options,
        device,
        preprocess_obs,
        model_dir,
        gif_interval,
        pause,
        args,
    ):
        self.log_wandb = args.log_wandb
        if self.log_wandb:
            wandb.init(project="LESSON")
            date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            default_model_name = "{}_{}_{}".format(
                args.env, args.algorithm, date
            )
            model_name = args.model or default_model_name
            wandb.run.name = model_name

        self.env = env
        self.eval_env = eval_env
        self.gif_interval = gif_interval
        self.model_dir = model_dir
        self.test_call_count = 0 
        self.pause = pause

        obs_space, _ = utils.get_obss_preprocessor(env.observation_space)
        include_mission = utils.check_run.include_mission(args.env)
        self.policy_network = DQN(obs_space, env.action_space, True, include_mission).to(device)
        self.target_network = DQN(obs_space, env.action_space, True, include_mission).to(device)

        self.memory = ReplayMemory(args.max_memory, preprocess_obs)

        utils.common_init.init(self, env=env, preprocess_obs=preprocess_obs, args=args, train_interval=10)
        utils.common_init.init_log(self, model_dir=model_dir)
        self.rnd_policy_network = DQN(obs_space, env.action_space, True, include_mission).to(device)
        self.rnd_target_network = DQN(obs_space, env.action_space, True, include_mission).to(device)
        self.rnd_network = RND(obs_space, 16, 64, device)
        utils.common_init.init_rnd(self, args=args)
        utils.common_init.init_optionQ(
            self,
            env=env,
            args=args,
            exploration_options=exploration_options,
        )
        # (this is old desc)
        # array of objects like {obs, actions} where obs is some env state 
        # and actions is count of actions taken in this state
        self.action_done_cache = [0 for i in range(self.n_actions)]
        self.explo_weights = [1.0 for i in range(self.n_actions)]

        self.local_action_cache = {}

    def collect_experiences(
        self,
        start_time,
        episode,
        num_frames,
        return_per_frame_,
        test_return_per_frame_,
    ):
        obs = self.env.reset()[0]
        preprocessed_obs = self.preprocess_obs([obs], device=self.device)
        episode_step = 0
        done = False
        option_termination = True

        log_loss, log_reward = [], []

        while not done and episode_step < self.max_episode_length:
            preprocessed_obs = self.preprocess_obs([obs], device=self.device)
            if option_termination:
                current_option = self.option_policy_network.select_option(preprocessed_obs, self.exploration_options, self.softmax_ww)
                self.w = random.randrange(self.n_actions)

            action, _ = utils.action.select_action_from_option(self, preprocessed_obs, None, current_option, obs)
            new_obs, reward, done, _, _ = self.env.step(action)
            new_preprocessed_obs = self.preprocess_obs([new_obs], device=self.device)

            reward = reward * 10
            reward_i = self.rnd_network.get_reward(new_preprocessed_obs).detach().item()

            done_mask = 0.0 if done else 1.0
            self.memory.add(
                {
                    "step": num_frames,
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "new_obs": new_obs,
                    "done": done_mask,
                    "reward_i": reward_i,
                    "option": current_option
                }
            )
            termination, sigmoid_termonations = self.option_policy_network.predict_option_termination(new_preprocessed_obs, current_option)
            option_termination = bool(Bernoulli(termination).sample().item())

            if num_frames % self.train_interval == 0 and len(self.memory) >= batch_size:
                collected_experience = self.memory.sample(batch_size)
                loss = self.train(collected_experience)
                if type(loss) is float:
                    log_loss.append(loss)
                option_loss = OptionQ.get_option_td_error(self, collected_experience=collected_experience)
                termination_loss, termination_error = OptionQ.get_termination_loss_batch(self, collected_experience=collected_experience)
                option_loss += termination_loss

                self.option_optimizer.zero_grad()
                option_loss.backward()
                self.option_optimizer.step()

            # print log
            log_reward.append(reward)
            if num_frames % self.log_interval == 0 and "rewards" in self.logs:
                utils.log.set_log(self, num_frames, start_time, episode, return_per_frame_, test_return_per_frame_,
                                    current_option, termination.item(), termination_error.item(), sigmoid_termonations)

            # test model
            self.test(num_frames, test_return_per_frame_)

            obs = new_obs
            episode_step += 1
            num_frames += 1

        logs = {"num_frames": num_frames, "rewards": log_reward, "loss": log_loss}
        self.logs = logs
        return logs

    def train(self, collected_experience):
        if self.learn_step_counter % self.update_target_per_train == 0:
            self.update_target_network()

        loss = DQN.train_model(
            online_net=self.policy_network,
            target_net=self.target_network,
            optimizer=self.optimizer,
            collected_experience=collected_experience,
            is_rnd=False,
        )
        DQN.train_model(
            online_net=self.rnd_policy_network,
            target_net=self.rnd_target_network,
            optimizer=self.rnd_optimizer,
            collected_experience=collected_experience,
            is_rnd=True,
            )

        new_obs = collected_experience["new_obs"]
        self.rnd_network.update(self.rnd_network.get_reward(new_obs))

        self.learn_step_counter += 1
        return loss.item()

    def update_target_network(self):
        print("Target network update".encode('utf8'))
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.rnd_target_network.load_state_dict(
            self.rnd_policy_network.state_dict()
        )
        self.option_target_network.load_state_dict(
            self.option_policy_network.state_dict()
        )

    def test(self, num_frames, test_return_per_frame_):
        if num_frames % self.test_interval == 0:
            self.test_call_count += 1
            print(f"test start @ num frames: {num_frames}".encode('utf8'))
            test_return = []
            for i in range(5):
                test_logs = self.test_collect_experiences(num_frames, i, save_gif=self.gif_interval > 0 and self.test_call_count % self.gif_interval == 0)
                test_return_per_episode = utils.synthesize(test_logs["rewards"])
                test_return.append(list(test_return_per_episode.values())[2])
            test_return_per_frame_.append(np.mean(test_return))

    def test_collect_experiences(self, num_frames, test_id, save_gif=False):
        obs = self.eval_env.reset()[0]
        done = False

        log_loss = []
        log_reward = []
        episode_step = 0
        frames = []  # Initialize an array to store frames
        while not done and episode_step < self.max_episode_length:
            episode_step += 1
            self.eval_env.render() # not sure if needed
            if save_gif:
                frames.append(np.moveaxis(self.eval_env.get_frame(), 2, 0))

            preprocessed_obs = self.preprocess_obs([obs], device=self.device)

            action, _ = utils.action.select_greedy_action(self, preprocessed_obs, None)
            new_obs, reward, done, _, _ = self.eval_env.step(action)
            log_reward.append(reward)
            obs = new_obs
        
        if save_gif:
            gif_dir = os.path.join(self.model_dir, "gifs")
            os.makedirs(gif_dir, exist_ok=True)
            gif_path = os.path.join(gif_dir, f"visualization_at_frame_{num_frames}_{test_id}.gif")
            write_gif(np.array(frames), gif_path, fps=1/self.pause)
            print(f"GIF saved at {gif_path}")

        logs = {"num_frames": None, "rewards": log_reward, "loss": log_loss}
        self.logs = logs
        return logs
    
    def save_model(self, filepath):
        state = {
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        state = torch.load(filepath, map_location=lambda storage, loc: storage)
        self.policy_network.load_state_dict(state['policy_network_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")   
