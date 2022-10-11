import gym
from dqn import *

env = gym.make("LunarLander-v2")

dqn = DQN(4)

dqn.play(env)
dqn.learn(env, 65000)
dqn.play(env)