import torch
import random
import numpy as np

def select_greedy_action(
    self,
    preprocessed_obs,
    hidden_states,
):
    new_hidden_states = hidden_states
    with torch.no_grad():
        if hidden_states and "q" in hidden_states:
            hidden_state = hidden_states["q"]
            Q, hidden_state = self.policy_network(preprocessed_obs, hidden_state)
            new_hidden_states = {"q": hidden_state, "rnd": hidden_states["rnd"]}
        else:
            Q = self.policy_network(preprocessed_obs)
        action = torch.argmax(Q).item()
    return action, new_hidden_states

def select_action_from_option(
    self,
    preprocessed_obs,
    hidden_states,
    current_option,
    obs
):
    new_hidden_states = hidden_states
    with torch.no_grad():
        if hidden_states and "q" in hidden_states:
            hidden_state = hidden_states["q"]
            hidden_state_rnd = hidden_states["rnd"]
            Q, hidden_state = self.policy_network(preprocessed_obs, hidden_state)
            Q_rnd, hidden_state_rnd = self.rnd_policy_network(
                preprocessed_obs, hidden_state_rnd
            )
            new_hidden_states = {"q": hidden_state, "rnd": hidden_state_rnd}
        else:
            Q = self.policy_network(preprocessed_obs)
            Q_rnd = self.rnd_policy_network(preprocessed_obs)
    
    self.action_count = []
    for cached in self.action_done_cache:
        if np.array_equal(cached["obs"], obs["image"]):
            self.action_count = cached["actions"]
    # that means we havent cached this state yet
    if len(self.action_count) == 0:
        handler = {"obs": obs["image"], "actions": [0 for i in range(self.n_actions)] }
        self.action_done_cache.append(handler)
        self.action_count = handler["actions"]

    if current_option == 0:
        self.type = "r"
        action = random.randrange(self.n_actions)
    elif current_option == 1:
        self.type = "z"
        action = self.w
    # elif current_option == 2:
    #     self.type = "rnd"
    #     action = torch.argmax(Q_rnd).item()
    elif current_option == 2:
        self.type = "c"
        explo_weights = [1.0 / (x + 1) for x in self.action_count]
        action = random.choices(range(len(explo_weights)), weights=explo_weights, k=1)[0]
    elif current_option == 3:
        self.type = "e"
        action = torch.argmax(Q).item()

    self.action_count[action] = self.action_count[action] + 1
    return action, new_hidden_states