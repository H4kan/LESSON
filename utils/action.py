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
    
    hash = calculate_state_hash(obs["image"][0])

    if hash not in self.local_action_cache:
        self.local_action_cache[hash] = [[0 for i in range(self.n_actions)],
                                         [1.0 for i in range(self.n_actions)]]

    if current_option == 0:
        self.type = "r"
        action = random.randrange(self.n_actions)
    elif current_option == 1:
        self.type = "z"
        action = self.w
    # elif current_option == 2:
    #     self.type = "rnd"
    #     action = torch.argmax(Q_rnd).item()
    # elif current_option == 2:
    #     self.type = "c"
    #     action = random.choices(range(len(self.explo_weights)), weights=self.explo_weights, k=1)[0]
    elif current_option == 2:
        self.type = "c"
        action = random.choices(range(self.n_actions), weights=self.local_action_cache[hash][1], k=1)[0]
    elif current_option == 3:
        self.type = "e"
        action = torch.argmax(Q).item()

    self.local_action_cache[hash][0][action] = self.local_action_cache[hash][0][action] + 1
    self.local_action_cache[hash][1][action] = 1.0 / (self.local_action_cache[hash][0][action] + 1)
    # self.action_done_cache[action] = self.action_done_cache[action] + 1
    # self.explo_weights[action] = 1.0 / (self.action_done_cache[action] + 1);
    return action, new_hidden_states

def calculate_state_hash(state):
    rowSum = 1
    colSum = 1
    for arr in state:
        rowSum = rowSum * max(1, sum(arr))
    for i in range(0, len(state[0])):
        handler = 0
        for arr in state:
            handler = handler + arr[i]
        colSum = colSum * max(1, handler)
    return colSum - rowSum
