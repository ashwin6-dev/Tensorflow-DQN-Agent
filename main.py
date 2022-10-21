import gym
from dqn import *

env = gym.make("LunarLander-v2")

dqn = DQN(4, build_dense_policy_nn())

dqn.learn(env, 70000)
dqn.play(env)