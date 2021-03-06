import torch
from gym import spaces
import gym_super_mario_bros

from adept.env._spaces import Space
from adept.preprocess.base.preprocessor import CPUPreprocessor, GPUPreprocessor
from adept.preprocess.ops import (
    CastToFloat,
    GrayScaleAndMoveChannel,
    ResizeToNxM,
    Divide,
    FrameStackCPU,
    FromNumpy,
)
from adept.env.base.env_module import EnvModule

from nes_py.wrappers import JoypadSpace
from mariorl.modules.preprocess import SkipFrame

MARIO_ENVS = ["SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0"]


class AdeptMarioEnv(EnvModule):

    args = {
        "frame_stack": True,
        "max_episode_length": 40000,
        "skip_rate": 4,
    }

    ids = MARIO_ENVS

    def __init__(self, env, do_frame_stack, skip_rate, max_episode_length):
        # Define the preprocessing operations to be performed on observation
        # CPU OPS
        cpu_ops = [
            FromNumpy("Box", "Box"),
            GrayScaleAndMoveChannel("Box", "Box"),
            ResizeToNxM(84, 84, "Box", "Box"),
        ]
        if do_frame_stack:
            cpu_ops.append(FrameStackCPU("Box", "Box", 4))
        cpu_preprocessor = CPUPreprocessor(
            cpu_ops,
            Space.from_gym(env.observation_space),
            Space.dtypes_from_gym(env.observation_space),
        )

        # GPU OPS
        gpu_preprocessor = GPUPreprocessor(
            [CastToFloat("Box", "Box"), Divide("Box", "Box", 255)],
            cpu_preprocessor.observation_space,
            cpu_preprocessor.observation_dtypes,
        )

        action_space = Space.from_gym(env.action_space)
        print("action space length: ", action_space["Discrete"][0])
        super(AdeptMarioEnv, self).__init__(
            action_space, cpu_preprocessor, gpu_preprocessor
        )

        self.gym_env = env
        self._gym_obs_space = env.observation_space

    @classmethod
    def from_args(cls, args, seed, **kwargs):
        env = gym_super_mario_bros.make(args.env)
        env = JoypadSpace(env, [["right"], ["right", "A"]])
        env = SkipFrame(env, skip=args.skip_rate)
        env._max_episode_steps = args.max_episode_length
        env.seed(seed)
        return cls(
            env, args.frame_stack, args.skip_rate, args.max_episode_length
        )

    def step(self, action):
        obs, reward, done, info = self.gym_env.step(
            self._wrap_action(action).item()
        )
        return self._wrap_observation(obs), reward, done, info

    def reset(self, **kwargs):
        self.cpu_preprocessor.reset()
        obs = self.gym_env.reset(**kwargs)
        return self._wrap_observation(obs)

    def close(self):
        self.gym_env.close()

    def __getattr__(self, item):
        return getattr(self.gym_env, item)

    def _wrap_observation(self, observation):
        space = self._gym_obs_space
        if isinstance(space, spaces.Box):
            return self.cpu_preprocessor({"Box": observation.copy()})
        elif isinstance(space, spaces.Discrete):
            # onehot encode net1d inputs
            longs = torch.from_numpy(observation)
            if longs.dim() > 2:
                raise ValueError(
                    "observation is not net1d, too many dims: "
                    + str(longs.dim())
                )
            elif len(longs.dim()) == 1:
                longs = longs.unsqueeze(1)
            one_hot = torch.zeros(observation.size(0), space.n)
            one_hot.scatter_(1, longs, 1)
            return self.cpu_preprocessor({"Discrete": one_hot.numpy()})
        elif isinstance(space, spaces.MultiBinary):
            return self.cpu_preprocessor({"MultiBinary": observation})
        elif isinstance(space, spaces.Dict):
            return self.cpu_preprocessor(
                {name: obs for name, obs in observation.items()}
            )
        elif isinstance(space, spaces.Tuple):
            return self.cpu_preprocessor(
                {idx: obs for idx, obs in enumerate(observation)}
            )
        else:
            raise NotImplementedError

    def _wrap_action(self, action):
        return action["Discrete"]
