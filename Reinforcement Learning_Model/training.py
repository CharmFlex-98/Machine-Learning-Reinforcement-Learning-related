import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from custom_environment import *


def create_model():
    model = models.resnet18(pretrained=False)
    fc_layer_num = model.fc.in_features
    custom_fc_layer = nn.Sequential(
        nn.Linear(fc_layer_num, 100),
        nn.BatchNorm1d(10),
        nn.ReLU(),
        nn.Linear(100, 4),
    )
    model.fc = custom_fc_layer

    return model


episodes = 1000
time_step = 1000
learning_rate=0.01
exploration_rate_decay=0.001
max_exploration_rate=1
min_exploration_rate=0.01

window_h = 500
window_w = 500
bird_icon = 'bird-average-bird-lifespans-thinkstock-155253666.jpg'
bird_icon_size = (60, 60)
wall_icon = 'wall1.jpg'
wall_icon_size = (150, 100)
food_icon = 'food.jpg'
food_icon_size = (50, 50)
env = CustomEnv(bird_icon, bird_icon_size, wall_icon, wall_icon_size, food_icon, food_icon_size)
env.reset()
plt.imshow(env.next_state)
plt.show()
for i in count():
    env.render('rgb_array')
    action = env.action_space.sample()
    state, _, _ = env.step(action)
    # plt.imshow(state)
    # plt.show()



