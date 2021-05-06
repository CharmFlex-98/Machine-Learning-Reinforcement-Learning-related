import torch
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y=[4, 9, 16, 25, 36, 49, 64, 81, 100, 121]

for index, num in enumerate(x):
    plt.cla()
    plt.plot(x[:index+1], y[:index+1])
    plt.ylim([0, 200])
    plt.xlim([0, 20])
    plt.pause(0.1)

for z in range(1000000):
    print(z)
    plt.pause(0.0000000000000000000000000000000001)

for index, num in enumerate(x):
    plt.cla()
    plt.plot(x[:index+1], y[:index+1])
    plt.ylim([0, 200])
    plt.xlim([0, 20])
    plt.pause(0.001)




