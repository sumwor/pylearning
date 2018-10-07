import numpy as np

required = ['inputs', 'actions', 'tmax', 'n_gradient', 'n_validation']
default  = {
    'Performance':           None,
    'N':                     100,
    'p0':                    0.1,  # connection probability of decision network
    'baseline_N':            100,
    'baseline_p0':           1,
    'lr':                    0.004,
    'baseline_lr':           0.004,
    'max_iter':              100000,
    'fix':                   [],
    'baseline_fix':          [],
    'target_reward':         np.inf,
    'mode':                  'continues',  # should be contineous?
    'network_type':          'gru',
    'baseline_network_type': 'gru',
    'R_ABORTED':             -1,
    'R_TERMINAL':            None,
    'abort_on_last_t':       True,
    'checkfreq':             50,
    'dt':                    10,
    'tau':                   100,
    'tau_reward':            np.inf,
    'var_rec':               0.01,
    'baseline_var_rec':      0.01,
    'L2_r':                  0,
    'baseline_L2_r':         0,
    'Win':                   1,
    'baseline_Win':          None,
    'bout':                  1,  #bias to one decision?
    'baseline_bout':         None,
    'Win_mask':              None,
    'baseline_Win_mask':     None,
    'rho':                   2,
    'baseline_rho':          2,
    'L1_Wrec':               0,
    'L2_Wrec':               0,
    'policy_seed':           1,
    'baseline_seed':         2
    }
