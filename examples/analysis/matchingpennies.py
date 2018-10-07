from __future__ import absolute_import, division

import os

import numpy as np

from pyrl          import fittools, runtools, tasktools, utils
from pyrl.figtools import apply_alpha, Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from scipy.signal import savgol_filter

# /////////////////////////////////////////////////////////////////////////////////////////

def single_trial(trialsfile, savefile):
    trials, A, R, M, perf = utils.load(trialsfile)
    randomTrial = np.random.randint(0,200)
    condition = trials[randomTrial]['offer']
    len_go = 200
    len_decision = 2000
    len_ITI = trials[randomTrial]['durations']['ITI'][1] - trials[randomTrial]['durations']['ITI'][0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rect_go = plt.Rectangle((0, 0), len_go, 1)
    rect_decision = plt.Rectangle((len_go,0), len_decision, 1)
    rect_ITI = plt.Rectangle((len_go+len_decision,0), len_ITI,1)
    ax.add_patch(rect_go)
    ax.add_patch(rect_decision)
    ax.add_patch(rect_ITI)
    for action in A:
        plt.vlines(x, ymin, ymax)
    filename = savefile + '/trial'+ str(randomTrial)  + '.png'
    print filename
    plt.savefig(filename, format='png')
    plt.show()

    plt.close()

def choice_pattern(trialsfile, offers, savefile, action,**kwargs):
    # Load trials
    #this is up to date

    trials, A, R, M, perf = utils.load(trialsfile)

    # ComChoice = []
    Choice = []
    Reward = []
    n_nondecision = 0
    for n, trial in enumerate(trials):
        if perf.choices[n] is None:
            n_nondecision += 1
            continue

        juice = trial['juice']
        offer = trial['offer']

        if perf.choices[n] == 'LEFT':
            C = -1
        elif perf.choices[n] == 'RIGHT':
            C = 1
        else:
            raise ValueError("invalid choice")
        if perf.corrects[n]:
            R = 1
        else:
            R = 0

        Choice.append(C)
        Reward.append(R)
    print("Non-decision trials: {}/{}".format(n_nondecision, len(trials)))

    # pL_by_offer = {}
    # for offer in B_by_offer:
    #     Bs = B_by_offer[offer]
    #     pB_by_offer[offer] = utils.divide(sum(Bs), len(Bs))
         # print(offer, pB_by_offer[offer])

    #-------------------------------------------------------------------------------------
    # Plot
    #-------------------------------------------------------------------------------------

    #if plot is None:
    #    return

    # ms       = kwargs.get('ms', 7)
    # rotation = kwargs.get('rotation', 60)

    #for i, offer in enumerate(offers):
    #    plot.plot(i, 100*pB_by_offer[offer], 'o', color='k', ms=ms)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.bar([x for x in range(1, len(Choice)+1)], [-1 if x == -1 else 0 for x in Choice], color='r', edgeColor='none')
    plt.bar([x for x in range(1, len(Choice)+1)], [1 if x == 1 else 0 for x in Choice], 1, color='b', edgeColor='none')
    plt.ylabel("Animal's choice")
    plt.yticks([-1, 0, 1], ('Left', ' ', 'Right'))
    # plt.yticklabels({'Left', 'Right'});
    plt.xlim(0, len(Choice))

    plt.subplot(2, 1, 2)
    average_reward = sum(Reward)/len(Choice)
    smooth = savgol_filter(Reward, 11, 1)
    plt.bar([x for x in range(1, len(Choice) + 1)], Reward, color='black', edgeColor='none')
    plt.plot([x for x in range(1, len(Choice) + 1)], smooth, color='red', linewidth=2)
    plt.ylabel("Reward")
    plt.yticks([0, 1], ('No reward', 'Reward'))
    plt.text((len(Choice)-50), 1.05, "Average reward: " + str(average_reward))
    # plt.yticklabels({'Left', 'Right'});
    plt.xlim(0, len(Choice))

    filename = savefile + '/' + action + '.png'
    print filename
    plt.savefig(filename, format='png')
    plt.show()

    plt.close()
#/////////////////////////////////////////////////////////////////////////////////////////

def do(action, args, config):
    """
    Manage tasks.

    """
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    #=====================================================================================

    if action == 'plot_trial':

        try:
            trials_per_condition = int(args[0])
        except:
            trials_per_condition = 1000
        model = config['model']
        pg = model.get_pg(config['savefile'], config['seed'], config['dt'])

        spec = model.spec
        juices = spec.juices
        offers = spec.offers
        n_conditions = spec.n_conditions
        n_trials = trials_per_condition * n_conditions

        print("{} trials".format(n_trials))
        task = model.Task()

        fig  = Figure(axislabelsize=10, ticklabelsize=9)
        plot = fig.add()

        plot_trial()
        performance(config['savefile'], plot)

        fig.save(path=config['figspath'], name='performance')
        fig.close()

    #=====================================================================================

    elif 'trials' in action:
        try:
            trials_per_condition = int(args[0])
        except IndexError:
            trials_per_condition = 100

        model = config['model']
        pg    = model.get_pg(config['savefile'], config['seed'], config['dt'])

        spec         = model.spec
        juices       = spec.juices
        offers       = spec.offers
        n_conditions = spec.n_conditions
        n_trials     = trials_per_condition * n_conditions

        print("{} trials".format(n_trials))
        task   = model.Task()
        trials = []
        for n in xrange(n_trials):
            k = tasktools.unravel_index(n, (len(juices), len(offers)))
            context = {
                'juice': juices[k.pop(0)],
                'offer': offers[k.pop(0)]
                }
            trials.append(task.get_condition(pg.rng, pg.dt, context))
        runtools.run(action, trials, pg, config['trialspath'])


    #=====================================================================================

    elif action == 'choice_pattern':

        trialsfile = runtools.behaviorfile(config['trialspath'])
        # print trialsfile
        # fig = Figure()

        # plot = fig.add()
        savefile = config['figspath']

        spec = config['model'].spec

        #print spec.offers
        choice_pattern(trialsfile, spec.offers, savefile, action)

        #plot.xlabel('Offer (\#B : \#A)')

        #plot.ylabel('Percent choice B')

        #plot.text_upper_left('1A = {}B'.format(spec.A_to_B), fontsize=10)



    #=====================================================================================

    elif action == 'sort':
        if 'value' in args:
            network = 'v'
        else:
            network = 'p'

        trialsfile = runtools.activityfile(config['trialspath'])
        sort(trialsfile, (config['figspath'], 'sorted'), network=network)

    #=====================================================================================

    elif action == 'statespace':
        trialsfile = runtools.activityfile(config['trialspath'])
        statespace(trialsfile, (config['figspath'], 'statespace'))


# for debug
offers = [(0, 1), (1, 0)]
trialsfile = "/home/hongli/scratch/work/pyrl/RLearning/matchingpennies/trials_behavior.pkl"
savefile ='/home/hongli/Documents/RLearning/work/figs/matchingpennies'
action = "choice_pattern"
choice_pattern(trialsfile, offers, savefile, action)
single_trial(trialsfile, savefile)