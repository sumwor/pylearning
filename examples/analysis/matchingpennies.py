from __future__ import absolute_import, division

import os

import numpy as np

from pyrl          import fittools, runtools, tasktools, utils
from pyrl.figtools import apply_alpha, Figure
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# /////////////////////////////////////////////////////////////////////////////////////////

def plot_trial(pg, m, init, init_b, rng, figspath, name):
    #context = {}
    #if 0 not in m.cohs:
    #    context['cohs'] = [0] + m.cohs
    #trial = m.generate_trial_condition(rng, context)

    U, Z, A, R, M, init, states_0, perf = pg.run_trials([trial], init=init)
    if pg.baseline_net is not None:
        (init_b, baseline_states_0, b,
         rpe) = pg.baseline_run_trials(U, A, R, M, init=init_b)
    else:
        b = None

    U = U[:,0,:]
    Z = Z[:,0,:]
    A = A[:,0,:]
    R = R[:,0]
    M = M[:,0]
    t = int(np.sum(M))

    w = 0.65
    h = 0.18
    x = 0.17
    dy = h + 0.05
    y0 = 0.08
    y1 = y0 + dy
    y2 = y1 + dy
    y3 = y2 + dy

    fig   = Figure(h=6)
    plots = {'observables': fig.add([x, y3, w, h]),
             'policy':      fig.add([x, y2, w, h]),
             'actions':     fig.add([x, y1, w, h]),
             'rewards':     fig.add([x, y0, w, h])}

    time        = trial['time']
    dt          = time[1] - time[0]
    act_time    = time[:t]
    obs_time    = time[:t] + dt
    reward_time = act_time + dt
    xlim        = (0, max(time))

    #-------------------------------------------------------------------------------------
    # Observables
    #-------------------------------------------------------------------------------------

    plot = plots['observables']
    plot.plot(obs_time, U[:t,0], 'o', ms=5, mew=0, mfc=Figure.colors('blue'))
    plot.plot(obs_time, U[:t,0], lw=1.25, color=Figure.colors('blue'),   label='Fixation')
    plot.plot(obs_time, U[:t,1], 'o', ms=5, mew=0, mfc=Figure.colors('orange'))
    plot.plot(obs_time, U[:t,1], lw=1.25, color=Figure.colors('orange'), label='Left')
    plot.plot(obs_time, U[:t,2], 'o', ms=5, mew=0, mfc=Figure.colors('purple'))
    plot.plot(obs_time, U[:t,2], lw=1.25, color=Figure.colors('purple'), label='Right')
    try:
        plot.plot(obs_time, U[:t,3], 'o', ms=5, mew=0, mfc=Figure.colors('green'))
        plot.plot(obs_time, U[:t,3], lw=1.25, color=Figure.colors('green'), label='Sure')
    except IndexError:
        pass

    plot.xlim(*xlim)
    plot.ylim(0, 1)
    plot.ylabel('Observables')

    coh = trial['left_right']*trial['coh']
    if coh < 0:
        color = Figure.colors('orange')
    elif coh > 0:
        color = Figure.colors('purple')
    else:
        color = Figure.colors('k')
    plot.text_upper_right('Coh = {:.1f}\%'.format(coh), color=color)

    props = {'prop': {'size': 7}, 'handlelength': 1.2,
             'handletextpad': 1.2, 'labelspacing': 0.8}
    plot.legend(bbox_to_anchor=(1.2, 0.8), **props)

    plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------
    # Policy
    #-------------------------------------------------------------------------------------

    plot = plots['policy']
    plot.plot(act_time, Z[:t,0], 'o', ms=5, mew=0, mfc=Figure.colors('blue'))
    plot.plot(act_time, Z[:t,0], lw=1.25, color=Figure.colors('blue'),
              label='Fixate')
    plot.plot(act_time, Z[:t,1], 'o', ms=5, mew=0, mfc=Figure.colors('orange'))
    plot.plot(act_time, Z[:t,1], lw=1.25, color=Figure.colors('orange'),
              label='Saccade LEFT')
    plot.plot(act_time, Z[:t,2], 'o', ms=5, mew=0, mfc=Figure.colors('purple'))
    plot.plot(act_time, Z[:t,2], lw=1.25, color=Figure.colors('purple'),
              label='Saccade RIGHT')
    try:
        plot.plot(act_time, Z[:t,3], 'o', ms=5, mew=0, mfc=Figure.colors('green'))
        plot.plot(act_time, Z[:t,3], lw=1.25, color=Figure.colors('green'),
                  label='Saccade SURE')
    except IndexError:
        pass

    plot.xlim(*xlim)
    plot.ylim(0, 1)
    plot.ylabel('Action probabilities')

    props = {'prop': {'size': 7}, 'handlelength': 1.2,
             'handletextpad': 1.2, 'labelspacing': 0.8}
    plot.legend(bbox_to_anchor=(1.27, 0.8), **props)

    plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------
    # Actions
    #-------------------------------------------------------------------------------------

    plot = plots['actions']
    actions = [np.argmax(a) for a in A[:t]]
    plot.plot(act_time, actions, 'o', ms=5, mew=0, mfc=Figure.colors('red'))
    plot.plot(act_time, actions, lw=1.25, color=Figure.colors('red'))
    plot.xlim(*xlim)
    yticklabels = ['Fixate', 'Saccade LEFT', 'Saccade RIGHT']
    if A.shape[1] == 4:
        yticklabels += ['Saccade sure']
    plot.yticklabels(yticklabels)
    plot.ylim(0, len(yticklabels)-1)
    plot.yticks(range(len(yticklabels)))

    plot.ylabel('Action')

    plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------
    # Rewards
    #-------------------------------------------------------------------------------------

    plot = plots['rewards']
    plot.plot(reward_time, R[:t], 'o', ms=5, mew=0, mfc=Figure.colors('red'))
    plot.plot(reward_time, R[:t], lw=1.25, color=Figure.colors('red'))

    # Prediction
    if b is not None:
        plot.plot(reward_time, b[:t], 'o', ms=5, mew=0, mfc=Figure.colors('orange'))
        plot.plot(reward_time, b[:t], lw=1.25, color=Figure.colors('orange'))

    plot.xlim(*xlim)
    plot.ylim(m.R_TERMINATE, m.R_CORRECT)
    plot.xlabel('Time (ms)')
    plot.ylabel('Reward')

    plot.highlight(0, m.iti)

    #-------------------------------------------------------------------------------------

    fig.save(path=figspath, name=name)
    fig.close()

    #-------------------------------------------------------------------------------------

    return init, init_b

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
