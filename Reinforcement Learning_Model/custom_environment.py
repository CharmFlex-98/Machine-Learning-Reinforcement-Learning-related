import numpy as np
import matplotlib.pyplot as plt
import random
from gym import Env, spaces
import time
import cv2
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.nn.functional as F


class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        self.batch_size = batch_size

    def push_memory_in(self, memory):
        if len(self.memory) < self.capacity:
            self.memory.append(memory)
        else:
            self.memory[self.push_count % self.capacity] = memory
        self.push_count += 1

    def take_sample(self):
        return random.sample(self.memory, self.batch_size)

    def can_provide_sample(self):
        return len(self.memory) >= self.batch_size


class CustomEnv(Env):
    def __init__(self, bird_icon, size1, wall_icon, size2, food_icon, size3):
        super(CustomEnv, self).__init__()
        self.observation_shape = (window_h, window_h, 3)
        self.action = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}
        self.action_space = spaces.Discrete(4, )
        self.canvas = np.full(self.observation_shape, fill_value=255)
        self.elements = []
        self.bird = Element('bird', bird_icon, size1, self.elements)
        self.wall = Element('wall', wall_icon, size2, self.elements)
        self.food = Element('food', food_icon, size3, self.elements)
        self.wall_up = False

        self.current_screen = None
        self.next_state = None
        self.reward = 0
        self.done = False
        self.just_starting = True

        self.current_step = 0

    def choose_action(self, state):
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * \
                           math.exp(-self.current_step * exploration_rate_decay)
        print(exploration_rate)
        if random.random() > exploration_rate:
            with torch.no_grad():
                Qmodel.eval()
                _action = Qmodel(state).argmax(1)
                Qmodel.train()

                return _action  # exploit
        else:
            return torch.tensor([self.action_space.sample()]).to(device)

    def step(self, action_index):
        action_reward = 0
        action_index = action_index.item()
        self.current_step += 1
        # bird
        #  or self.bird.x == self.bird.max_x or self.bird.y == self.bird.min_y or \
        #         self.bird.y == self.bird.max_y:                            # if out of range then deduct mark

        if self.action[action_index] == 'left':
            if self.bird.x == self.bird.min_x:  # check if bound to walls
                action_reward -= 30
            self.bird.move(-5, 0)
            action_reward -= 5

        elif self.action[action_index] == 'right':
            if self.bird.x == self.bird.max_x:
                action_reward -= 30
            self.bird.move(5, 0)
            action_reward += 10

        elif self.action[action_index] == 'up':
            if self.bird.y == self.bird.min_y:
                action_reward -= 30
            self.bird.move(0, -5)

        elif self.action[action_index] == 'down':
            if self.bird.y == self.bird.max_y:
                action_reward -= 30
            self.bird.move(0, 5)

        # wall
        self.wall_moving_direction()
        if self.wall_up:
            self.wall.move(0, -3)
        else:
            self.wall.move(0, 3)

        action_reward, self.done = self.detect_collision(action_reward)

        self.update_canvas()
        self.render('rgb')

        return self.next_state, torch.tensor([action_reward]).float().to(device)

    def wall_moving_direction(self):
        if self.wall.y + self.wall.icon_h > window_h - 5:
            self.wall_up = True
        if self.wall.y < 5:
            self.wall_up = False

    def detect_collision(self, _reward):
        done = False
        if self.wall.x - self.bird.icon_w <= self.bird.x <= self.wall.x + self.wall.icon_w and \
                self.wall.y - self.bird.icon_h <= self.bird.y <= self.wall.y + self.wall.icon_h:
            done = True
            _reward -= 200
        if self.food.x - self.bird.icon_w <= self.bird.x <= self.food.x + self.food.icon_w and \
                self.food.y - self.bird.icon_h <= self.bird.y <= self.food.y + self.food.icon_h:
            done = True
            _reward += 1000
            print('win!')

        return _reward, done

    def update_canvas(self):
        self.canvas = np.full(self.observation_shape, fill_value=255)
        for element in self.elements:
            self.canvas[element.y:element.y + element.icon_h, element.x:element.x + element.icon_w] = element.icon

        self.next_state = self.get_state()

        Text = 'Reward: {}'.format(self.reward)
        self.canvas = cv2.putText(self.canvas, Text, (5, 5), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 125, 0))  # mutable

    def render(self, mode='human'):
        plt.clf()
        if mode == 'human':
            cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
            cv2.imshow('Canvas', self.canvas)
            cv2.waitKey(10)
        else:
            plt.imshow(self.canvas)
            plt.pause(0.001)

    def reset(self):
        self.just_starting = True
        self.done = False
        self.reward = 0
        self.canvas = np.full(self.observation_shape, fill_value=255)
        for element in self.elements:
            if element.name == 'bird':
                element.x = random.randint(0, 40)
                element.y = random.randint(10, 90)
            elif element.name == 'wall':
                element.x = 50
                element.y = int(self.observation_shape[0] / 2 - element.icon_h / 2)
            else:
                element.x = 100
                element.y = random.randint(10, 100)
            self.canvas[element.y:element.y + element.icon_h, element.x:element.x + element.icon_w] = element.icon
        self.next_state = self.get_state()

        Text = 'Reward: {}'.format(self.reward)
        self.canvas = cv2.putText(self.canvas, Text, (5, 5), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 125, 0))

    def get_state(self):
        if self.just_starting or self.done:
            self.current_screen = np.copy(self.canvas)
            black_screen = np.zeros_like(self.canvas)
            return torch.tensor(black_screen).permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            s1 = self.current_screen
            s2 = np.copy(self.canvas)
            self.current_screen = s2

            return torch.tensor(s2 - s1).permute(2, 0, 1).unsqueeze(0).to(device)


class Motion:
    def __init__(self, name, max_x, max_y, min_x=0, min_y=0):
        self.name = name
        self.x = 0
        self.y = 0
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def move(self, velocity_x, velocity_y):
        self.x += velocity_x
        self.y += velocity_y
        self.clamp()

    def clamp(self):
        self.x = min(max(self.x, self.min_x), self.max_x)
        self.y = min(max(self.y, self.min_y), self.max_y)


class Element(Motion):
    def __init__(self, name, icon, icon_size, element_list):
        self.icon = cv2.imread(icon)
        self.icon = cv2.cvtColor(self.icon, cv2.COLOR_BGR2RGB)
        self.icon_w = icon_size[1]
        self.icon_h = icon_size[0]
        self.element = element_list
        super(Element, self).__init__(name, max_x=window_w - self.icon_w, max_y=window_h - self.icon_h)
        self.resize()
        self.push_into_element()

    def resize(self):
        self.icon = cv2.resize(self.icon, (self.icon_w, self.icon_h))

    def push_into_element(self):
        self.element.append(self)


class Qvalue:
    @staticmethod
    def get_current(states, actions):
        return Qmodel(states).gather(1, actions.unsqueeze(-1))  # need calculate gradient later on

    @staticmethod
    def get_next(next_states):
        black_screen_boolean = next_states.flatten(1).max(1)[0].eq(0).type(
            torch.bool)  # torch.tensor([True, False, ...])
        non_black_screen_boolean = (black_screen_boolean == False)
        non_black_states = next_states[non_black_screen_boolean]  # 4D tensor
        Batch_size = next_states.shape[0]
        values = torch.zeros(Batch_size).to(device)
        values[non_black_screen_boolean] = Tmodel(non_black_states).max(1)[0].detach()  # no need to calculate gradient

        return values


class custom_model(nn.Module):
    def __init__(self):
        super(custom_model, self).__init__()

        self.conv1 = nn.Conv2d(3, 100, 3, 2)
        self.batchnorm1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 50, 3, 2)
        self.batchnorm2 = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 10, 3, 2)
        self.batchnorm3 = nn.BatchNorm2d(10)

        self.linear1 = nn.Linear(10 * 13 * 13, 100)
        self.batchnorm4 = nn.BatchNorm1d(100)
        self.linear2 = nn.Linear(100, 4)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.batchnorm4(self.linear1(x)))
        output = self.linear2(x)

        return output


def extract_tensor(batches):
    batch = memory(*zip(*batches))
    t1 = torch.cat(batch.state)  # torch.tensor([a, b, c, d, ...])
    t2 = torch.cat(batch.action)  # torch.tensor([a, b, c, d, ...])
    t3 = torch.cat(batch.next_state)  # torch.size([batch_size, 3, h, w])
    t4 = torch.cat(batch.reward)  # torch.size([batch_size, 3, h, w])

    return t1, t2, t3, t4


def plot(_episode, _reward_history, period):
    pass
    # training_graph.clf()
    # plt.plot(range(period, _episode + period, period), _reward_history)
    # training_graph.pause(0.001)


'''
--------------------training--------------------------------------
'''


# training_graph = plt.figure()

def train():
    periodic_reward = 0
    for episode in range(episodes):
        print('{}{}'.format('running episode', episode))
        env.reset()  # env.next_state is black screen, env.current_screen is env.canvas
        env.just_starting = False
        state = env.next_state.float()  # initialize current screen (black screen)
        for step in range(time_step):
            # plt.imshow(state.squeeze().permute(1, 2, 0).to('cpu'))
            # plt.show()
            action = env.choose_action(state)
            next_state, reward = env.step(action)  # show the latest frame at the same time
            next_state = next_state.float()
            env.reward += reward.item()
            periodic_reward += reward.item()
            replay_memory.push_memory_in(memory(state, action, next_state, reward))
            state = next_state
            if replay_memory.can_provide_sample():
                memory_batch = replay_memory.take_sample()
                states, actions, next_states, rewards = extract_tensor(memory_batch)
                current_Qvalue = Qvalue.get_current(states, actions)
                next_Qvalue = Qvalue.get_next(next_states)
                target_Qvalue = rewards + (discount_factor * next_Qvalue)

                loss = loss_function(current_Qvalue, target_Qvalue.unsqueeze(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if env.done:
                break

        if episode % Tmodel_update_period == 0:
            Tmodel.load_state_dict(Qmodel.state_dict())

        if episode % graph_episode_period == 0:
            reward_history.append(periodic_reward / graph_episode_period)
            periodic_reward = 0
            # plot(episode, reward_history, graph_episode_period)
            print(reward_history)

    torch.save(Qmodel.state_dict(), 'learned.pt')
    print('save!')

    env.close()


def test():
    Qmodel.load_state_dict(torch.load('learned.pt'))
    Qmodel.eval()
    for episode in range(20):
        env.reset()
        env.just_starting = False
        state = env.next_state.float()

        for step in range(time_step):
            with torch.no_grad():
                action = Qmodel(state).argmax(1)
            next_step, _ = env.step(action)
            next_step = next_step.float()
            state = next_step

            if env.done:
                break


if __name__ == '__main__':
    training = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    episodes = 1500
    time_step = 100
    learning_rate = 0.01
    batch_size = 256
    discount_factor = 0.95
    exploration_rate_decay = 0.00008
    max_exploration_rate = 1
    min_exploration_rate = 0.01

    window_h = 112
    window_w = 112
    bird_icon = 'bird-average-bird-lifespans-thinkstock-155253666.jpg'
    bird_icon_size = (15, 15)
    wall_icon = 'wall1.jpg'
    wall_icon_size = (40, 30)
    food_icon = 'food.jpg'
    food_icon_size = (10, 10)

    memory = namedtuple('memory',
                        ('state', 'action', 'next_state', 'reward'))  # these are all needed for finding Qvalue

    Qmodel = custom_model()
    Tmodel = custom_model()
    Qmodel.to(device)
    Tmodel.to(device)

    Tmodel.load_state_dict(Qmodel.state_dict())
    Tmodel.eval()
    Tmodel_update_period = 10
    optimizer = optim.Adam(Qmodel.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss(reduction='mean')
    graph_episode_period = 10

    env = CustomEnv(bird_icon, bird_icon_size, wall_icon, wall_icon_size, food_icon, food_icon_size)
    replay_memory = ReplayMemory(capacity=10000, batch_size=batch_size)

    reward_history = []
    real_time_training_window = plt.figure()

    if training:
        train()
    else:
        test()
