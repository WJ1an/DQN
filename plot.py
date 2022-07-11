import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np


def plot_smooth(log_reward):
    y = savgol_filter(log_reward, 91, 3)
    plt.plot(range(len(y)), y)
    plt.show()


data = np.load("log/DQN/reward.pkl.npy")
plot_smooth(data)
