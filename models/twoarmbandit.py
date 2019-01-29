"""
matching pennies task, based on


"""
from __future__ import division

import numpy as np
from random import random

from pyrl import tasktools

import scipy.stats

# Inputs
inputs = tasktools.to_map('Choice', 'Reward')
"""
input situations:
1. go cue
2. choice : 0(left), 1(right)
3. reward : 0, 1
"""


# Actions
actions = tasktools.to_map('CHOOSE-LEFT', 'CHOOSE-RIGHT')

# Trial conditions
#determine what to put here later
#A_to_B       = 2.2
#juices       = [('A', 'B'), ('B', 'A')]
juices        = ['water']
offers        = [(0.7, 0.1), (0.1, 0.7)]
n_conditions = len(juices) * len(offers)


# Training
n_gradient   = n_conditions
n_validation = 50*n_conditions

# Input noise
sigma = np.sqrt(2*100*0.001)

# Durations
go     = 200    #go cue played for 200 ms
#reward = 3000   #time for acquiring the reward: 3000 ms
ITI_mean = 3000
ITI_min = 1000      # truncated exponential distribution
ITI_max = 5000      # truncated exponential distribution (define in tasktools.py already)
#pause   = 3000      #time-out
reward    = 3000  #waitlick period: 2000 ms
tmax         = go + reward + ITI_max

# Rewards
R_ABORTED = 0
R_water   = 1

# Increase initial policy -> baseline weights
baseline_Win = 10

# Input scaling
def scale(x):
    return x/5


def get_condition(rng, dt,blockID, context={}):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------
    #print "running here"
    ITI = context.get('ITI')
    if ITI is None:
        ITI = tasktools.truncated_exponential(rng, dt, ITI_min, ITI_min, ITI_max)
    #print ITI
    durations = {
        'go':        (0, go),
        'reward':  (go, go + reward),
        #'reward':    (go + decision, go + decision + reward),
        'ITI':       (go + reward,  go + reward + ITI),
        'tmax':      tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    #-------------------------------------------------------------------------------------
    # Trial
    #-------------------------------------------------------------------------------------

    juice = context.get('juice')
    if juice is None:
        juice = juices

    offer = context.get('offer')
    if offer is None:

        # offer = tasktools.choice(rng, offers)
        offer = offers[blockID]
    #juiceL, juiceR = juice
    #nB, nA = offer
    nL,nR = offer
    #if juiceL == 'A':
    #    nL, nR = nA, nB
    #else:
    #    nL, nR = nB, nA

    return {
        'durations': durations,
        'time':      time,
        'epochs':    epochs,
        'juice':     juices,
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
    #time_out = 0
    #ITI_repeat = 0

    if t-1 in epochs['go']:
        if a in [actions['CHOOSE-LEFT'], actions['CHOOSE-RIGHT']]:
            status['jump'] = True
            status['t_choice'] = t-1

            #juiceL, juiceR = trial['juice']

            # add a funtion to determine which trial it is

            nLeft, nRight = trial['offer']

            randL = random()
            randR = random()

            if randL < nLeft:
                rLeft = R_water
            else:
                rLeft = 0

            if randR < nRight:
                rRight = R_water
            else:
                rRight = 0

           # if juiceL == 'A':
           #     rL, rR = rA, rB
           # else:
           #     rL, rR = rB, rA

            if a == actions['CHOOSE-LEFT']:

                status['choice'] = 0  # for left
                reward = rLeft
                status['correct'] = rLeft
            elif a == actions['CHOOSE-RIGHT']:

                status['choice'] = 1  # for right
                status['correct'] = rRight
                reward = rRight

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    # noise later

    u = np.zeros(len(inputs))
    u[inputs['Choice']] = rng.normal(scale=sigma) / np.sqrt(dt)
    u[inputs['Reward']] = rng.normal(scale=sigma) / np.sqrt(dt)
    if t in epochs['reward']:
        #juiceL, juiceR = trial['juice']
        if 'choice' in status.keys():
            u[inputs['Choice']] = scale(status['choice']) + rng.normal(scale=sigma)/np.sqrt(dt)
            u[inputs['Reward']] = scale(reward) + rng.normal(scale=sigma)/np.sqrt(dt)
            print "======================================================================="
            print "u", u
        #why scale?
        #u[inputs['N-L']] = scale(trial['nL']) + rng.normal(scale=sigma)/np.sqrt(dt)
        #u[inputs['N-R']] = scale(trial['nR']) + rng.normal(scale=sigma)/np.sqrt(dt)

    #-------------------------------------------------------------------------------------

    return u, reward, status

def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.6



#for debug
#rng = np.random.RandomState(1234)
#trial = get_condition(rng,dt=10)