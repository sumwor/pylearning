"""
matching pennies task, based on


"""
from __future__ import division

import numpy as np

from pyrl import tasktools

import scipy.stats

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
#reward = 3000   #time for acquiring the reward: 3000 ms
ITI_mean = 3000
ITI_min = 1000      # truncated exponential distribution
ITI_max = 5000      # truncated exponential distribution (define in tasktools.py already)
#pause   = 3000      #time-out
decision     = 2000  #waitlick period: 2000 ms
tmax         = go + decision + ITI_max

# Rewards
R_ABORTED = 0
R_water   = 1

# Increase initial policy -> baseline weights
baseline_Win = 10

# Input scaling
def scale(x):
    return x/5


def get_condition(rng, dt, context={}, choiceHis=[], rewardHis=[], trial_count=0):
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
        'decision':  (go, go + decision),
        #'reward':    (go + decision, go + decision + reward),
        'ITI':       (go + decision,  go + decision + ITI),
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
        pred_choice = binomial_test(choiceHis, rewardHis, trial_count)
        #offer = tasktools.choice(rng, offers)
        offer = offers[pred_choice - 2]
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
    if t-1 in epochs['ITI'] :
        if a != actions['HOLD']:
            status['continue'] = False
            reward = R_ABORTED
    elif t-1 in epochs['decision']:
        if a in [actions['CHOOSE-LEFT'], actions['CHOOSE-RIGHT']]:
            #status['jump'] = True
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

                status['choice'] = 'LEFT'
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
    if t in epochs['decision']:
        #juiceL, juiceR = trial['juice']
        if 'choice' in status.keys():
            if status['choice'] == 'LEFT':
                if status['correct']:
                    u[inputs['L-R']] = 1
                else:
                    u[inputs['L-N']] = 1
            else:
                if status['correct']:
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

    return p_decision >= 0.99 and p_correct >= 0.5

def binomial_test(c_his, r_his, iter):
    #def a function to do the matching_pennies choices test
    #c_his: choice history; r_his: reward history; iter: number of iteration (500 trials as a block)
    alpha = 0.05
    p_min = 0.5
    trials_include = iter % 500
    if trials_include > 5:
        for N in range(0,5):
            left,right = choice_counting(c_his[-trials_include:], r_his[-trials_include:], N)
            pValue = scipy.stats.binom_test(right, left+right, 0.5, alternative='two-sided')

            if pValue < alpha:
                if abs(0.5 - right/(left+right)) > abs(0.5 - p_min):
                    p_min == right/(left+right)
                    #print p_min

    #get choice
    if np.random.rand() < p_min:
        next_choice = 2
    else:
        next_choice = 3

    return next_choice


def choice_counting(choiceHistory, rewardHistory, num):
    leftCount = 0
    rightCount = 0

    if num == 0:
        for i in range(len(choiceHistory)):
            if choiceHistory[i] == 2:
                leftCount += 1
            else:
                rightCount += 1
    else:
        comb = choiceHistory[-num:]
        combRew = rewardHistory[-num:]
        for i in range(len(choiceHistory) - num):
            if choiceHistory[i:i + num] == comb and rewardHistory[i:i+num] == combRew:
                if choiceHistory[i + num] == 2:
                    leftCount += 1
                else:
                    rightCount += 1


    return (leftCount, rightCount)

#for debug
#rng = np.random.RandomState(1234)
#trial = get_condition(rng,dt=10)