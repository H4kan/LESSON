import datetime
import random
import utils
# import wandb
import numpy as np
from torch.distributions import Bernoulli

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
        args,
    ):
        # self.log_wandb = args.log_wandb
        # if self.log_wandb:
        #     wandb.init(project="LESSON")
        #     date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        #     default_model_name = "{}_{}_{}".format(
        #         args.env, args.algorithm, date
        #     )
        #     model_name = args.model or default_model_name
        #     wandb.run.name = model_name

        self.env = env
        self.eval_env = eval_env

        obs_space, _ = utils.get_obss_preprocessor(env.observation_space)
        include_mission = utils.check_run.include_mission(args.env)
        self.policy_network = DQN(obs_space, env.action_space, True, include_mission).to(device)
        self.target_network = DQN(obs_space, env.action_space, True, include_mission).to(device)

        # self.policy_network.load_state_dict(utils.get_model_state(model_dir))
        # if hasattr(preprocess_obs, "vocab"):
        #     preprocess_obs.vocab.load_vocab(utils.get_vocab(model_dir))

        self.memory = ReplayMemory(args.max_memory, preprocess_obs)

        utils.common_init.init(self, env=env, preprocess_obs=preprocess_obs, args=args, train_interval=10)
        # self.optimizer.load_state_dict(utils.get_o)
        
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

        # status = utils.get_status(model_dir)    
        # if "optimizer_state" in status:
        #     self.optimizer.load_state_dict(status["optimizer_state"])

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

            action, _ = utils.action.select_action_from_option(self, preprocessed_obs, None, current_option)
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
        print("Target network update")
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.rnd_target_network.load_state_dict(
            self.rnd_policy_network.state_dict()
        )
        self.option_target_network.load_state_dict(
            self.option_policy_network.state_dict()
        )

    def test(self, num_frames, test_return_per_frame_):
        if num_frames % self.test_interval == 0:
            print(f"test start @ num frames: {num_frames}")
            test_return = []
            for _ in range(5):
                test_logs = self.test_collect_experiences()
                test_return_per_episode = utils.synthesize(test_logs["rewards"])
                test_return.append(list(test_return_per_episode.values())[2])
            test_return_per_frame_.append(np.mean(test_return))

    def test_collect_experiences(self):
        obs = self.eval_env.reset()[0]
        done = False

        log_loss = []
        log_reward = []
        episode_step = 0
        while not done and episode_step < self.max_episode_length:
            episode_step += 1
            preprocessed_obs = self.preprocess_obs([obs], device=self.device)

            action, _ = utils.action.select_greedy_action(self, preprocessed_obs, None)
            new_obs, reward, done, _, _ = self.eval_env.step(action)
            log_reward.append(reward)
            obs = new_obs

        logs = {"num_frames": None, "rewards": log_reward, "loss": log_loss}
        self.logs = logs
        return logs
