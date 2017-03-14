import copy
import numpy as np
from matplotlib import cm

action_space = ['stick', 'hit']

def draw():
    card_number = np.random.randint(1, 11)
    card_color = np.random.choice(['red', 'black'], 1, replace=True, p=[1/3, 2/3])

    return {
        'value': card_number,
        'color': card_color[0]
    }

def step(current_state=None, action='hit'):
    if not action in action_space:
        raise Exception('action must be one of the following: ' + ''.join(action_space))

    reward = 0
    end = False
    if current_state == None:
        # Init state
        return (
            {
                'dealer': np.random.randint(1, 11),
                'user': np.random.randint(1, 11),
                'end': False
            }
            , reward
        )

    next_state = copy.deepcopy(current_state)
    if action == 'hit':
        new_card = draw()
        if new_card['color'] == 'black':
            next_state['user'] += new_card['value']
        else:
            next_state['user'] -= new_card['value']

        if next_state['user'] > 21 or next_state['user'] < 1:
            # You got busted
            reward = -1
            next_state['end'] = True
    else:
        while(next_state['dealer'] >= 1 and next_state['dealer'] < 17):
            new_card = draw()
            if new_card['color'] == 'black':
                next_state['dealer'] += new_card['value']
            else:
                next_state['dealer'] -= new_card['value']

        if next_state['dealer'] > 21 or next_state['dealer'] < 1:
            # Dealer's got busted
            reward = 1
            next_state['end'] = True
        elif next_state['dealer'] < next_state['user']:
            # Dealer's got beaten
            reward = 1
            next_state['end'] = True
        else:
            # Draw
            next_state['end'] = True

    return (next_state, reward)

def getStateId(state):
    return (state['dealer'] - 1) * 21 + (state['user'] - 1)

def getActionId(action):
    return 0 if action == 'stick' else 1

def playEpisode(epsilon, Qs=None, N_s=None, N_0=None, thetas=None):
    history = []
    current_state, reward = step()
    while(current_state['end'] != True):
        history.append(reward)
        history.append(current_state)
        current_state_id = getStateId(current_state)
        # We select an action with an epsilon-greedy action-value based policy
        if Qs != None:
            if np.random.rand() >= epsilon[current_state_id]:
                    max = np.argmax(Qs[current_state_id])
                    action = action_space[max]
            else:
                action = np.random.choice(action_space)

            # Increment state and epsilon counter
            N_s[current_state_id] = N_s[current_state_id] + 1
            epsilon[current_state_id] = N_0 / (N_0 + N_s[current_state_id])
        else:
            if np.random.rand() >= epsilon:
                q_stick = phi(current_state, 'stick').dot(thetas)
                q_hit = phi(current_state, 'hit').dot(thetas)
                max = np.argmax([q_stick, q_hit])
                action = action_space[max]
            else:
                action = np.random.choice(action_space)

        # Update history
        history.append(action)
        # print('taking action: ' + action)

        # Retrieve the next_state and actions from the environment
        next_state, reward = step(current_state, action)
        # print('Next state: ', next_state, reward)

        current_state = copy.deepcopy(next_state)
    # If the game ends, we just append the reward in the history
    history.append(reward)

    return (epsilon, N_s, history)

def phi(state, action):
    dealer_phi = np.expand_dims(np.array([
        1 if state['dealer'] >= 1 and state['dealer'] <= 4 else 0,
        1 if state['dealer'] >= 4 and state['dealer'] <= 7 else 0,
        1 if state['dealer'] >= 7 and state['dealer'] <= 10 else 0,
    ]), 0)
    player_phi =  np.expand_dims(np.array([
        1 if state['user'] >= 1 and state['user'] <= 6 else 0,
        1 if state['user'] >= 4 and state['user'] <= 9 else 0,
        1 if state['user'] >= 7 and state['user'] <= 12 else 0,
        1 if state['user'] >= 10 and state['user'] <= 15 else 0,
        1 if state['user'] >= 13 and state['user'] <= 18 else 0,
        1 if state['user'] >= 16 and state['user'] <= 21 else 0,
    ]), 0)
    action_phi =  np.expand_dims(np.array([
        1 if action == 'hit' else 0,
        1 if action == 'stick' else 0,
    ]), 0)
    phi_tmp = np.dot(np.array(dealer_phi).T, np.array(player_phi)).flatten()
    phi = np.dot(np.expand_dims(phi_tmp, 0).T, np.array(action_phi)).flatten()

    return phi

def buildQs(thetas):
    Qs = []
    for i in range(1, 11):
        for j in range(1, 22):
            s = { 'dealer': i, 'user': j }
            q_stick = phi(s, 'stick').dot(thetas)
            q_hit = phi(s, 'hit').dot(thetas)
            Qs.append([q_stick, q_hit])

    return Qs

def updatePlot(ax, Qs):
    # Plotting
    X = []
    Y = []
    Z = []
    for i in range(10):
        X.append([])
        Y.append([])
        Z.append([])
        for j in range(11, 21):
            X[i].append(i + 1)
            Y[i].append(j + 1)
            Z[i].append(np.max(Qs[i * 21 + j]))

    ax.clear()
    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Sum")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)

    return