import copy, json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from easy21 import action_space, step, playEpisode, updatePlot, getStateId, getActionId

def update(i):
    global N_s, N_s_a, Qs, alpha, epsilon

    print('Step %d, running %d simulations' % (i, nb_episodes_per_graph_update))
    for i in range(nb_episodes_per_graph_update):
        epsilon, N_s, history = playEpisode(epsilon, Qs, N_s, N_0)      

        MCupdate(history)
        
    updatePlot(ax, Qs)
    
    return

def MCupdate(history):
    global N_s, N_s_a, Qs, alpha, epsilon

    for i in range(0, len(history) - 1, 3):
        state_id = getStateId(history[i + 1])
        action_id = getActionId(history[i + 2])
        G = 0 # undiscounted reward sample
        for j in range(i + 3, len(history), 3):
            G += history[j]

        Qs[state_id][action_id] += alpha[state_id][action_id] * (G - Qs[state_id][action_id])
        N_s_a[state_id][action_id] += 1
        alpha[state_id][action_id] = 1 / N_s_a[state_id][action_id]

    return

if __name__ == "__main__":
    nb_episodes = 400
    nb_episodes_per_graph_update = 2500

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    N_0 = 1e5
    N_s = np.ones([10 * 21])
    N_s_a = np.ones([10 * 21, 2])
    Qs = np.zeros([10 * 21, 2]) # Number of action-states pair possible
    alpha = 1 / N_s_a
    epsilon = N_0 / (N_0 + N_s)

    anim = FuncAnimation(fig, update, frames=nb_episodes, interval=200)

    anim.save('results/mcplot.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    with open('results/mc-qstar.json', 'w') as outfile:
        json.dump(Qs.tolist(), outfile)
