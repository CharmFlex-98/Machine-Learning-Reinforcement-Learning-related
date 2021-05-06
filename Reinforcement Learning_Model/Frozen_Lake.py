import numpy as np
import gym
from IPython.display import clear_output
import time
import random

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.01

discount_rate = 0.99

exploration_rate_threshold = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

env = gym.make("FrozenLake-v0")
action_size = env.action_space.n
state_size = env.observation_space.n
q_table = np.zeros((state_size, action_size))

rewards_all_episode=[]

for episode in range(num_episodes):
    state = env.reset()
    done = False
    reward_current_episode = 0

    for step in range(max_steps_per_episode):
        exploration_rate = random.uniform(0, 1)
        if exploration_rate < exploration_rate_threshold:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(q_table[state, :])  # exploit

        new_state, reward, done, info = env.step(action)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                 learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        state = new_state
        reward_current_episode += reward

        if done:
            break

    exploration_rate_threshold = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * \
                                 np.exp(-exploration_decay_rate * episode)

    rewards_all_episode.append(reward_current_episode)

rewards_per_1000=np.split(np.array(rewards_all_episode), 10)
count=1000
for reward in rewards_per_1000:
    average_reward=float(sum(reward)/1000)
    print('{} : {}'.format(count, average_reward))
    count+=1000

print(q_table)

# # rendering result
# for episode in range(3):
#     state = env.reset()
#     done=False
#     print('### episode',episode,'###')
#     time.sleep(1)
#
#     for step in range(max_steps_per_episode):
#         clear_output(wait=True)
#         env.render()
#         time.sleep(0.3)
#
#         action = np.argmax(q_table[state, :])
#         new_state, reward, done, info = env.step(action)
#
#         if done:
#             clear_output(wait=True)
#             env.render()
#             if reward == 1:
#                 print("****You reached the goal!****")
#                 time.sleep(3)
#             else:
#                 print("****You fell through a hole!****")
#                 time.sleep(3)
#                 clear_output(wait=True)
#             break
#
#         state=new_state
#
# env.close()

