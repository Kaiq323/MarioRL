import typing
import random
from typing import Dict, Tuple
from adept.agent import AgentModule
from adept.scripts.local import parse_args, main
if typing.TYPE_CHECKING:
    import torch
    from adept.network import NetworkModule
    from adept.rewardnorm import RewardNormModule
from collections import deque
import numpy as np
from mariorl.modules.adept_mario_net import AdeptMarioNet
from mariorl.modules.adept_mario_replay import AdeptMarioReplay

class AdeptMarioAgent(AgentModule):
    # You will be prompted for these when training script starts
    args = {"example_arg1": True, "example_arg2": 5}

    def __init__(self, reward_normalizer, action_space, spec_builder, state_dim):
        super(AdeptMarioAgent, self).__init__(
            reward_normalizer, action_space,
        )
        self.state_dim = state_dim
        self.action_dim = action_space['Discrete'][0]
        self.save_dir = "~/Documents/marioRL/mariorl/logs/"
        self.memory = AdeptMarioReplay.from_args(spec_builder)
        self.batch_size = 32
        self.use_cuda = torch.cuda.is_available()
        self.gamma = 0.9
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burn_in = 1e4 #min experiences before training
        self.learn_every = 3 #no. experiences between updates to Q_online
        self.sync_every = 1e4 #no. of experiences between Q_target and Q_online sync

        self.net = AdeptMarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.save_every = 5e5

    @classmethod
    def from_args(
        cls,
        args,
        reward_normalizer: RewardNormModule,
        action_space: Dict[str, Tuple[int, ...]],
        spec_builder,
        **kwargs
    ):
        pass

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s, a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_action_exp(
            self, predictions, internals, obs, available_actions
    ):
        raise NotImplementedError

    def learn_step(self, updater, network, next_obs, internals):
        #normalize rewards


        #Sample from memory
        state, next_state
        pass