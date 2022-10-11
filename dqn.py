import numpy as np
import tensorflow as tf
import random
import keras.backend as K
from matplotlib import pyplot as plt

def HuberLoss(mask_value, clip_delta):
  def f(y_true, y_pred):
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta
    mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    masked_squared_error = 0.5 * K.square(mask_true * (y_true - y_pred))
    linear_loss  = mask_true * (clip_delta * K.abs(error) - 0.5 * (clip_delta ** 2))
    huber_loss = tf.where(cond, masked_squared_error, linear_loss)
    return K.sum(huber_loss) / K.sum(mask_true)
  f.__name__ = 'masked_huber_loss'
  return f

class DQN:
    def __init__(self, action_n):
        self.action_n = action_n
        self.policy = self.build_model()
        self.target = self.build_model()
        self.replay = []
        self.max_replay_size = 10000

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(self.action_n, activation="linear"),
        ])

        model.compile(loss=HuberLoss(0, 1), optimizer=tf.keras.optimizers.Adam(0.0001))

        return model

    def play_episode(self, env, epsilon, max_timesteps):

        obs = env.reset()
        rewards = 0
        steps = 0

        for _ in range(max_timesteps):
            rand = np.random.uniform(0, 1)

            if rand <= epsilon:
                action = env.action_space.sample()
            else:
                actions = self.policy(np.array([obs])).numpy()
                action = np.argmax(actions)

            new_obs, reward, done, _ = env.step(action)
            if len(self.replay) == self.max_replay_size:
                self.replay = self.replay[1:]
                
            self.replay.append([obs, action, reward, new_obs, done])
            rewards += reward
            obs = new_obs
            steps += 1

            yield steps, rewards

            if done:
                env.close()
                break


    def learn(self, env, timesteps, train_every = 5, update_target_every = 25, show_every_episode = 4, batch_size = 64, discount = 0.8, min_epsilon = 0.05, min_reward=150):
        max_episode_timesteps = 1000
        episodes = 1
        epsilon = 1
        decay = np.e ** (np.log(min_epsilon) / (timesteps))
        steps = 0

        episode_list = []
        rewards_list = []

        while steps < timesteps:
            for ep_len, rewards in self.play_episode(env, epsilon, max_episode_timesteps):
                epsilon *= decay
                steps += 1


                if steps % train_every == 0 and len(self.replay) > batch_size:
                    batch = random.sample(self.replay, batch_size)
                    obs = np.array([o[0] for o in batch])
                    new_obs = np.array([o[3] for o in batch])

                    curr_qs = self.policy(obs).numpy()
                    future_qs = self.target(new_obs).numpy()

                    for row in range(len(batch)):
                        action = batch[row][1]
                        reward = batch[row][2]
                        done = batch[row][4]

                        if not done:
                            curr_qs[row][action] = reward + discount * np.max(future_qs[row])
                        else:
                            curr_qs[row][action] = reward
            
                    self.policy.fit(obs, curr_qs, batch_size=batch_size, verbose=0)
                
                if steps % update_target_every == 0 and len(self.replay) > batch_size:
                    self.target.set_weights(self.policy.get_weights())

            episodes += 1

            if episodes % show_every_episode == 0:
                print ("epsiode: ", episodes)
                print ("explore rate: ", epsilon)
                print ("episode reward: ", rewards)
                print ("episode length: ", ep_len)
                print ("timesteps done: ", steps)

                episode_list.append(episodes)
                rewards_list.append(rewards)

                if rewards > min_reward:
                    self.policy.save(f"policy-model-{rewards}")
        
        self.policy.save("policy-model-final")
        plt.plot(episode_list, rewards_list)
        plt.show()


    def play(self, env):

        for _ in range(10):
            obs = env.reset()
            done = False

            while not done:
                rand = np.random.uniform(0, 1)

                actions = self.policy(np.array([obs])).numpy()
                action = np.argmax(actions)

                obs, _, done, _ = env.step(action)
                env.render()
