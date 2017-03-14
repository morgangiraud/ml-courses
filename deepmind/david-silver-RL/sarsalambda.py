import copy, json, codecs
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from easy21 import action_space, step, playEpisode, updatePlot, getStateId, getActionId

def update(i):
    global N_s, N_s_a, Qs, alpha, epsilon

    print('Step %d, lambda %f, running %d simulations' % (i, lambda_value, nb_episodes_per_graph_update))
    for i in range(nb_episodes_per_graph_update):
        epsilon, N_s, history = playEpisode(epsilon, Qs, N_s, N_0)      

        sarsaLambdaUpdate(history)
        if lambda_value == 0:
            MSE_0.append(np.mean(np.square(Qstar - Qs)))
        if lambda_value == 1:
            MSE_1.append(np.mean(np.square(Qstar - Qs)))
        
    updatePlot(ax, Qs)
    
    return

def getQLambda(lambda_value, history, Qs, current_step):
    nb_rewards = (len(history) // 3 + 1) - current_step // 3 - 1
    qs_lambda = [0] * nb_rewards
    n_step_return = 0

    for j in range(current_step + 3, len(history), 3):
        # print('j: %d' % j)
        for k in range(n_step_return, len(qs_lambda)):
            # print('k: %d, n_step_return: %d' % (k, n_step_return))
            qs_lambda[k] = qs_lambda[k] + history[j]
            if k == n_step_return and j != len(history) -1:
                other_state_id = getStateId(history[j + 1])
                other_action_id = getActionId(history[j + 2])
                qs_lambda[k] = qs_lambda[k] + Qs[other_state_id][other_action_id]
        n_step_return += 1
        # print(qs_lambda)
        
    for i in range(len(qs_lambda)):
        qs_lambda[i] = np.power(lambda_value, i) * qs_lambda[i]
    # print(qs_lambda)

    if lambda_value == 1:
        q_lambda = np.sum(qs_lambda)
    else:
        q_lambda = (1 - lambda_value) * np.sum(qs_lambda)

    return q_lambda

def sarsaLambdaUpdate(history):
    global N_s, N_s_a, Qs, alpha, epsilon

    for i in range(0, len(history) - 1, 3):
        state_id = getStateId(history[i + 1])
        action_id = getActionId(history[i + 2])
        
        q_lambda = getQLambda(lambda_value, history, Qs, i)

        Qs[state_id][action_id] = Qs[state_id][action_id] + alpha[state_id][action_id] * (q_lambda - Qs[state_id][action_id])
        N_s_a[state_id][action_id] = N_s_a[state_id][action_id] + 1
        alpha[state_id][action_id] = 1 / N_s_a[state_id][action_id]

    return

if __name__ == "__main__":
    # Load Monte carlo estimates of Q star
    Qstar = np.array(json.loads(open('results/mc-qstar.json', 'r').read()))
    
    nb_episodes = 1
    nb_episodes_per_graph_update = 1000

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    lambdas = [i / 10 for i in range(11)]
    MSEs = []
    MSE_0 = []
    MSE_1 = []
    for lambda_value in lambdas:
        N_0 = 1e5
        N_s = np.ones([10 * 21])
        N_s_a = np.ones([10 * 21, 2])
        Qs = np.zeros([10 * 21, 2]) # Number of action-states pair possible
        alpha = 1 / N_s_a
        epsilon = N_0 / (N_0 + N_s)
        update(0)
        fig.savefig('results/slplot-lambda' + str(lambda_value) + '.png')

        MSEs.append(np.mean(np.square(Qstar - Qs)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambdas, MSEs)
    fig.savefig('results/slplot-mse.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, nb_episodes_per_graph_update + 1), MSE_0)
    fig.savefig('results/slplot-mse_0.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, nb_episodes_per_graph_update + 1), MSE_1)
    fig.savefig('results/slplot-mse_1.png')


