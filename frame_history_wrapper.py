from gym import Wrapper
from gym import spaces
from collections import deque

import numpy as np

class FrameHistoryWrapper(Wrapper):
    '''
    Gym wrapper implementing frame history, as published
    by e.g. Mnih et al. in 'Playing Atari with deep reinforcement learning'.

    Returns a FrameBuffer instance holding the past n frames.

    Credits to openai baselines - the implementation is basically the same
    '''
    def __init__(self, env, hl=4):
        assert env is not None
        super(FrameHistoryWrapper, self).__init__(env)

        # Set `hl`, init `frames` deque, override `observation_space`
        self._hl = hl
        self.frames = deque([], maxlen=hl)
        os = list(env.observation_space.shape)
        os = [hl] + os
        self.observation_space = spaces.Box(low=0, high=255, shape=tuple(os))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_state(), reward, done, info

    def _reset(self):
        obs = self.env.reset()
        for _ in range(self._hl):
            self.frames.append(obs)
        return self._get_state()

    def _get_state(self):
        assert len(self.frames) == self._hl
        return list(self.frames)
