"""
matching pennies task, based on


"""
from __future__ import division

import numpy as np

from pyrl import tasktools

# Inputs
inputs = tasktools.to_map('GO', 'L-R', 'R-R', 'L-N', 'R-N')
"""
input situations:
1. go cue
2. left reward
3. right reward
4. left no reward
5. right no reward
"""


# Actions
actions = tasktools.to_map('HOLD', 'CHOOSE-LEFT', 'CHOOSE-RIGHT')

# Trial conditions
#determine what to put here later
#A_to_B       = 2.2
#juices       = [('A', 'B'), ('B', 'A')]
juices        = ['water']
offers        = [(0, 1), (1, 0)]
n_conditions = len(juices) * len(offers)


# Training
n_gradient   = n_conditions
n_validation = 50*n_conditions

# Input noise
sigma = np.sqrt(2*100*0.001)

# Durations
go     = 200    #go cue played for 200 ms
reward = 3000   #time for acquiring the reward: 3000 ms
ITI_min = 1000      # truncated exponential distribution
ITI_max = 5000      # truncated exponential distribution (define in tasktools.py already)
#pause   = 3000      #time-out
decision     = 2000  #waitlick period: 2000 ms
tmax         = go + decision + reward  + ITI

# Rewards
R_ABORTED = 0
R_water   = 0.1

# Increase initial policy -> baseline weights
baseline_Win = 10

# Input scaling
def scale(x):
    return x/5

def get_condition(rng, dt, context={}):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    ITI = context.get('ITI')
    if ITI is None:
        ITI = tasktools.truncated_exponential(rng, dt, 3, ITI_min, ITI_max)

    durations = {
        'go':        (0, go),
        'decision':  (go, go + decision),
        'reward':    (go + decision, go + decision + reward),
        'ITI':       (go + decision + reward, go + decision + reward + ITI)
        'tmax':      tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    #-------------------------------------------------------------------------------------
    # Trial
    #-------------------------------------------------------------------------------------

    juice = context.get('juice')
    if juice is None:
        juice = tasktools.choice(rng, juices)

    offer = context.get('offer')
    if offer is None:
        offer = tasktools.choice(rng, offers)

    #juiceL, juiceR = juice
    #nB, nA = offer

    #if juiceL == 'A':
    #    nL, nR = nA, nB
    #else:
    #    nL, nR = nB, nA

    return {
        'durations': durations,
        'time':      time,
        'epochs':    epochs,
        'water':     water,
        'offer':     offer,
        'nL':        nL,
        'nR':        nR
        }

def get_step(rng, dt, trial, t, a):
    #-------------------------------------------------------------------------------------
    # Reward
    #-------------------------------------------------------------------------------------

    #find the trial running program later

    epochs = trial['epochs']
    status = {'continue': True}
    reward = 0
    time_out = 0
    ITI_repeat = 0
    if t-1 in epochs['ITI'] :
        if a != actions['HOLD'] and ITI_repeat < 5:
            ITI_repeat += 1
    elif t-1 in epochs['decision']:
        if a in [actions['CHOOSE-LEFT'], actions['CHOOSE-RIGHT']]:
            status['decision_stop'] = True
            status['t_choice'] = t-1

            #juiceL, juiceR = trial['juice']


            #add a funtion to determine which trial it is

            nLeft, nRight = trial['offer']
            rLeft     = nLeft * R_water
            rRight     = nRight * R_water

           # if juiceL == 'A':
           #     rL, rR = rA, rB
           # else:
           #     rL, rR = rB, rA

            if a == actions['CHOOSE-LEFT']:

                tatus['choice'] = 'LEFT'
                status['correct'] = (rLeft >= rRight)
                reward = rLeft
            elif a == actions['CHOOSE-RIGHT']:

                status['choice'] = 'RIGHT'
                status['correct'] = (rRight >= rLeft)
                reward = rRight

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    u = np.zeros(len(inputs))
    if t in epochs['go']:
        u[inputs['GO']] = 1
    if t in epochs['reward']:
        #juiceL, juiceR = trial['juice']
        if statue['choice'] == 'LEFT':
            if status['correct'] == True:
                u[inputs['L-R']] = 1
            else:
                u[inputs['L-N']] = 1
        else:
            if status['correct'] == True:
                u[inputs['R-R']] = 1
            else:
                u[inputs['R-N']] = 1

        #why scale?
        #u[inputs['N-L']] = scale(trial['nL']) + rng.normal(scale=sigma)/np.sqrt(dt)
        #u[inputs['N-R']] = scale(trial['nR']) + rng.normal(scale=sigma)/np.sqrt(dt)

    #-------------------------------------------------------------------------------------

    return u, reward, status

def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.95
