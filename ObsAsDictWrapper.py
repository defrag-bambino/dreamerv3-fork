import gym
from gym.spaces import Box, Dict
import numpy as np

class ObsAsDictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Dict({
            "image": Box(low=0, high=255, shape=env.observation_space[0].shape, dtype=np.uint8),
            "vector": env.observation_space[1]
            })
        

    def observation(self, obs):
        # rescale the image from 0-1 to 0-255 and convert to uint8
        image = obs[0] * 255
        image = image.astype(np.uint8)
        # pack into a dict and return
        return {"image": image, "vector": obs[1]}