import typing
import random
from typing import Dict, Tuple
from adept.agent import AgentModule
from adept.scripts.local import parse_args, main
if typing.TYPE_CHECKING:
    import torch
    from adept.network import NetworkModule
    from adept.rewardnorm import RewardNormModule
from collections import OrderedDict
import numpy as np
from mariorl.modules.adept_mario_net import AdeptMarioNet
from mariorl.modules.adept_mario_replay import AdeptMarioReplay

class AdeptMarioAgent(AgentModule):
    # You will be prompted for these when training script starts
    args = {"example_arg1": True, "example_arg2": 5}

    def __init__(self, reward_normalizer, action_space, spec_builder,
                 exp_size, exp_min_size, exp_update_rate, rollout_len,
                 discount, nb_env, return_scale):
        super(AdeptMarioAgent, self).__init__(
            reward_normalizer, action_space,
        )
        self._exp_cache = AdeptMarioReplay(spec_builder,exp_size, exp_min_size,
                                           rollout_len, exp_update_rate)



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

    def _get_qvals_from_pred(self, preds):
        return preds

    def _action_from_q_values(self, q_vals):
        return q_vals.argmax(dim=-1, keepdim=True)

    def _get_action_values(self, q_vals, action, batch_size=0):
        return q_vals.gather(1, action)

    def _values_to_tensor(self, values):
        return torch.cat(values, dim=1)

    def compute_action_exp(self, predictions, internals, obs,
                           available_actions):


    def learn_step(self, updater, network, next_obs, internals):
        #normalize rewards


        #Sample from memory
        state, next_state
        pass