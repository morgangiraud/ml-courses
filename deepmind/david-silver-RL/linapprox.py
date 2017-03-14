import copy, json, codecs
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from easy21 import action_space, step, playEpisode, updatePlot, phi, buildQs

def update(i):
    global thetas

    print('Step %d, lambda %f, running %d simulations' % (i, lambda_value, nb_episodes_per_graph_update))
    for i in range(nb_episodes_per_graph_update):
        _, _, history = playEpisode(epsilon, thetas=thetas)      

        sarsaLambdaUpdate(history)
        if lambda_value == 0:
            MSE_0.append(np.mean(np.square(Qstar - buildQs(thetas))))
        if lambda_value == 1:
            MSE_1.append(np.mean(np.square(Qstar - buildQs(thetas))))
        
    updatePlot(ax, buildQs(thetas))
    
    return

def sarsaLambdaUpdate(history):
    global thetas

    for i in range(0, len(history) - 1, 3):
        phi_vec = phi(history[i + 1], history[i + 2])
        q = phi_vec.dot(thetas)
        
        q_lambda = getQLambda(lambda_value, history, thetas, i)

        thetas +=  alpha * (q_lambda - q) * phi_vec

    return

def getQLambda(lambda_value, history, thetas, current_step):
    nb_rewards = (len(history) // 3 + 1) - current_step // 3 - 1
    qs_lambda = [0] * nb_rewards
    n_step_return = 0

    for j in range(current_step + 3, len(history), 3):
        for k in range(n_step_return, len(qs_lambda)):
            qs_lambda[k] = qs_lambda[k] + history[j]
            if k == n_step_return and j != len(history) -1:
                q = phi(history[j + 1], history[j + 2]).dot(thetas)
                qs_lambda[k] = qs_lambda[k] + q
        n_step_return += 1
        
    for i in range(len(qs_lambda)):
        qs_lambda[i] = np.power(lambda_value, i) * qs_lambda[i]

    if lambda_value == 1:
        q_lambda = np.sum(qs_lambda)
    else:
        q_lambda = (1 - lambda_value) * np.sum(qs_lambda)

    return q_lambda


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
        thetas = np.random.standard_normal(36)
        alpha = 0.01
        epsilon = 0.05
        update(0)
        fig.savefig('results/linslplot-lambda' + str(lambda_value) + '.png')

        MSEs.append(np.mean(np.square(Qstar - buildQs(thetas))))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambdas, MSEs)
    fig.savefig('results/linslplot-mse.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 1001), MSE_0)
    fig.savefig('results/linslplot-mse_0.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 1001), MSE_1)
    fig.savefig('results/linslplot-mse_1.png')


