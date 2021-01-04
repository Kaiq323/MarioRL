import torch
from pathlib import Path
import datetime
from gym.wrappers import FrameStack
#NES Emulator
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from mariorl.modules.preprocess import SkipFrame, GrayScaleObservation, \
    ResizeObservation
from mariorl.modules.agent import Mario
from mariorl.modules.logger import MetricLogger
import numpy as np
np.seterr(over='ignore')
#SETTING UP ENV
env =  gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

#Limit action space to
# 0. walk right
# 1. walk left
env = JoypadSpace(env, [["right"], ["right", "A"]])
env.reset()
next_state, reward, done, info = env.step(action=0)

#Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
print("length of action space: ", env.action_space.n)
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 40000
print(f"Running for {episodes} episodes")
# for e in range(episodes):
#     state = env.reset()
#
#     #Play the game!
#     while True:
#
#         #run agent on the state
#         action = mario.act(state)
#
#         #Agent performs action
#         next_state, reward, done, info = env.step(action)
#
#         #Remember
#         mario.cache(state, next_state, action, reward, done)
#
#         #Learn
#         q, loss = mario.learn()
#
#         #Logging
#         logger.log_step(reward, loss, q)
#
#         #Update State
#         state = next_state
#
#         #Check if end of game
#         if done or info["flag_get"]:
#             break
#
#     logger.log_episode()
#
#     if e % 20 == 0:
#         logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)