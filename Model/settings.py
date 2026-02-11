""" JSON PARAMETERS HERE (+DEFAULTS) """
# Simulation
duration = 1.e3 # (ms) simulation duration 
sim_dt = 0.025 # (ms) time step
sim_v_init = -65 # (mV)
sim_seed = 42
rho = 300 # medium resistivity (ohm cm)
sim_celsius = 35

# population sizes per area | [Pyr, BC, OLM] or [Gran, BC, HIPP] in DG
N_EC = [] # default: [50, 6, 2]
N_DG = [] # default: [100, 3, 1]
N_CA3 = [] # default: [26, 3, 1]
N_CA1 = [] # default: [100, 9, 3]
N_SUB = [] # default: [50, 6, 2]

# N_all = None

# population noise levels per area | [Pyr, BC, OLM] or [Gran, BC, HIPP] in DG
sigma_EC = [] # default: 
sigma_DG = [] # default: 
sigma_CA3 = [] # default: 
sigma_CA1 = [] # default: [1.e-2, 1.e-2, 2.5e-3]
sigma_SUB = [] # default:
# sigma_all = None

mean_EC = [] # default: 
mean_DG = [] # default: 
mean_CA3 = [] # default: 
mean_CA1 = [] # default: [0., 0., 0.]
mean_SUB = [] # default:
# mean_all = None

tau_EC = [] # default: 
tau_DG = [] # default: 
tau_CA3 = [] # default: 
tau_CA1 = [] # default: [25, 25, 25]
tau_SUB = [] # default:
# tau_all = None

seed_noise = 0

# synaptic distance per area | [Pyr, BC, OLM] or [Gran, BC, HIPP] in DG
syn_dist_EC = []
syn_dist_DG = []
syn_dist_CA3 = []
syn_dist_CA1 = [] # default: [500, 470, 1057.695]
syn_dist_SUB = []

# TODO : synaptic distance between areas
syn_dist_CA3_CA1 = []

# connection weights per area | [[Pyr-Pyr, Pyr-BC, Pyr-OLM], [BC-Pyr, BC-BC, BC-OLM], [OLM-Pyr, OLM-BC, OLM-OLM]]
                                # or [[Gran-Gran, Gran-BC, Gran-HIPP], [BC-Gran, BC-BC, BC-HIPP], [HIPP-Gran, HIPP-BC, HIPP-HIPP]] in DG
w_EC = [[],[], []] 
w_DG = [[],[], []] 
w_CA3 = [[],[], []] 
w_CA1 = [[],[], []] # default: [[0., 0.0045, 0.0009], [0.36, 0.0018, 0.], [0., 0., 0.]]
w_SUB = [[],[], []]

# synaptic types [rise, decay, reversal potential] and properties
syn_threshold = -20 # (mV)
syn_delay = 1 # (ms)

syn_exc = [] # default: [0.5, 3., 0]
syn_inh = [] # default: [1., 8., -75]

# connection weights inter areas
w_CA3_CA1 = []

# input
input_status = False
input_target = None # CA3/CA1
input_type = None # oscillatory/ramp
input_rate = 0 # frequency if oscillatory input
input_amp = 0 # mA
input_onset = 0 # ms
input_duration = 0 # ms

# extracellular stimulation
stim_status = False
stim_pos = [] # [x, y, z] coordinates of stimulation
stim_dur = 1 # (ms)
stim_onset = 100 # (ms)
stim_amp = [] # (mA)
stim_electrode = None # monopolar/bipolar
stim_type = None # single_pulse/train_pulses
stim_waveform = None # monophasic/biphasic
stim_freq = None # frequency if train of pulses
stim_pulse_width = None # ms
stim_interphase = 0 # ms time between two phases if biphasic stim
ATTACHED__ = 0

# Reproducibility settings
timestamp = None
git_branch = None
git_hash = None
git_short_hash = None

def init(data):
    """ This is used to set the global variables according to the JSON file parameters """

    # Neuronal population sizes > [Pyr, BC, OLM] or [Gran, BC, HIPP]
    global N_EC, N_DG, N_CA3, N_CA1, N_SUB #, N_all
#     N_EC = [data['areas']['EC']['Pyramidal']['N'], data['areas']['EC']['Basket']['N'], data['areas']['EC']['OLM']['N']]
#     N_DG = [data['areas']['DG']['Granule']['N'], data['areas']['DG']['Basket']['N'], data['areas']['DG']['HIPP']['N']]
    N_CA3 = [data['areas']['CA3']['Pyramidal']['N'], data['areas']['CA3']['Basket']['N'], data['areas']['CA3']['OLM']['N']]
    N_CA1 = [data['areas']['CA1']['Pyramidal']['N'], data['areas']['CA1']['Basket']['N'], data['areas']['CA1']['OLM']['N']]
#     N_SUB = [data['areas']['Sub']['Pyramidal']['N'], data['areas']['Sub']['Basket']['N'], data['areas']['Sub']['OLM']['N']]
#     N_all = [N_EC, N_DG, N_CA3, N_CA1, N_SUB]

    # Population noise
    global sigma_EC, sigma_DG, sigma_CA3, sigma_CA1, sigma_SUB #, sigma_all
    # sigma_EC = [data['areas']['EC']['Pyramidal']['noise']['sigma'], data['areas']['EC']['Basket']['noise']['sigma'], data['areas']['EC']['OLM']['noise']['sigma']]
    # sigma_DG = [data['areas']['DG']['Granule']['noise']['sigma'], data['areas']['DG']['Basket']['noise']['sigma'], data['areas']['DG']['HIPP']['noise']['sigma']]
    sigma_CA3 = [data['areas']['CA3']['Pyramidal']['noise']['sigma']] #, data['areas']['CA3']['Basket']['noise']['sigma'], data['areas']['CA3']['OLM']['noise']['sigma']]
    sigma_CA1 = [data['areas']['CA1']['Pyramidal']['noise']['sigma'], data['areas']['CA1']['Basket']['noise']['sigma'], data['areas']['CA1']['OLM']['noise']['sigma']]
    # sigma_SUB = [data['areas']['Sub']['Pyramidal']['noise']['sigma'], data['areas']['Sub']['Basket']['noise']['sigma'], data['areas']['Sub']['OLM']['noise']['sigma']]
    # sigma_all = [sigma_EC, sigma_DG, sigma_CA3, sigma_CA1, sigma_SUB]

    global mean_EC, mean_DG, mean_CA3, mean_CA1, mean_SUB #, mean_all
    # mean_EC = [data['areas']['EC']['Pyramidal']['noise']['mean'], data['areas']['EC']['Basket']['noise']['mean'], data['areas']['EC']['OLM']['noise']['mean']]
    # mean_DG = [data['areas']['DG']['Granule']['noise']['mean'], data['areas']['DG']['Basket']['noise']['mean'], data['areas']['DG']['HIPP']['noise']['mean']]
    mean_CA3 = [data['areas']['CA3']['Pyramidal']['noise']['mean']] #, data['areas']['CA3']['Basket']['noise']['mean'], data['areas']['CA3']['OLM']['noise']['mean']]
    mean_CA1 = [data['areas']['CA1']['Pyramidal']['noise']['mean'], data['areas']['CA1']['Basket']['noise']['mean'], data['areas']['CA1']['OLM']['noise']['mean']]
    # mean_SUB = [data['areas']['Sub']['Pyramidal']['noise']['mean'], data['areas']['Sub']['Basket']['noise']['mean'], data['areas']['Sub']['OLM']['noise']['mean']]
    # mean_all = [mean_EC, mean_DG, mean_CA3, mean_CA1, mean_SUB]

    global tau_EC, tau_DG, tau_CA3, tau_CA1, tau_SUB #, tau_all
    # tau_EC = [data['areas']['EC']['Pyramidal']['noise']['tau'], data['areas']['EC']['Basket']['noise']['tau'], data['areas']['EC']['OLM']['noise']['tau']]
    # tau_DG = [data['areas']['DG']['Granule']['noise']['tau'], data['areas']['DG']['Basket']['noise']['tau'], data['areas']['DG']['HIPP']['noise']['tau']]
    tau_CA3 = [data['areas']['CA3']['Pyramidal']['noise']['tau']] #, data['areas']['CA3']['Basket']['noise']['tau'], data['areas']['CA3']['OLM']['noise']['tau']]
    tau_CA1 = [data['areas']['CA1']['Pyramidal']['noise']['tau'], data['areas']['CA1']['Basket']['noise']['tau'], data['areas']['CA1']['OLM']['noise']['tau']]
    # tau_SUB = [data['areas']['Sub']['Pyramidal']['noise']['tau'], data['areas']['Sub']['Basket']['noise']['tau'], data['areas']['Sub']['OLM']['noise']['tau']]
    # tau_all = [tau_EC, tau_DG, tau_CA3, tau_CA1, tau_SUB]

    global seed_noise
    seed_noise = data['simulation']['seed_noise']

    # Synaptic distance intra-area [Pyr, BC, OLM] or [Granule, BC, HIPP]
    global syn_dist_EC, syn_dist_DG, syn_dist_CA3, syn_dist_CA1, syn_dist_SUB
#     syn_dist_EC = [data['connectivity']['intra']['EC']['syn_distance']['Pyramidal'], data['connectivity']['intra']['EC']['syn_distance']['Basket'], data['connectivity']['intra']['EC']['syn_distance']['OLM']]
#     syn_dist_DG = [data['connectivity']['intra']['DG']['syn_distance']['Granule'], data['connectivity']['intra']['DG']['syn_distance']['Basket'], data['connectivity']['intra']['DG']['syn_distance']['HIPP']]
#     syn_dist_CA3 = [data['connectivity']['intra']['CA3']['syn_distance']['Pyramidal'], data['connectivity']['intra']['CA3']['syn_distance']['Basket'], data['connectivity']['intra']['CA3']['syn_distance']['OLM']]
    syn_dist_CA1 = [data['connectivity']['intra']['CA1']['syn_distance']['Pyramidal'], data['connectivity']['intra']['CA1']['syn_distance']['Basket'], data['connectivity']['intra']['CA1']['syn_distance']['OLM']]
#     syn_dist_SUB = [data['connectivity']['intra']['Sub']['syn_distance']['Pyramidal'], data['connectivity']['intra']['Sub']['syn_distance']['Basket'], data['connectivity']['intra']['Sub']['syn_distance']['OLM']]

    # Connection weights intra area [[Pyr-Pyr, Pyr-BC, Pyr-OLM], [BC-Pyr, BC-BC, BC-OLM], [OLM-Pyr, OLM-BC, OLM-OLM]]
    global w_EC, w_DG, w_CA3, w_CA1, w_SUB
    # w_EC = [[data['connectivity']['intra']['EC']['weight']['Pyramidal']['Pyramidal'], data['connectivity']['intra']['EC']['weight']['Pyramidal']['Basket'], data['connectivity']['intra']['EC']['weight']['Pyramidal']['OLM']],
    #         [data['connectivity']['intra']['EC']['weight']['Basket']['Pyramidal'], data['connectivity']['intra']['EC']['weight']['Basket']['Basket'], data['connectivity']['intra']['EC']['weight']['Basket']['OLM']],
    #         [data['connectivity']['intra']['EC']['weight']['OLM']['Pyramidal'], data['connectivity']['intra']['EC']['weight']['OLM']['Basket'], data['connectivity']['intra']['EC']['weight']['OLM']['OLM']]]
    
    # w_DG = [[data['connectivity']['intra']['DG']['weight']['Granule']['Granule'], data['connectivity']['intra']['DG']['weight']['Granule']['Basket'], data['connectivity']['intra']['DG']['weight']['Granule']['HIPP']],
    #         [data['connectivity']['intra']['DG']['weight']['Basket']['Granule'], data['connectivity']['intra']['DG']['weight']['Basket']['Basket'], data['connectivity']['intra']['DG']['weight']['Basket']['HIPP']],
    #         [data['connectivity']['intra']['DG']['weight']['HIPP']['Granule'], data['connectivity']['intra']['DG']['weight']['HIPP']['Basket'], data['connectivity']['intra']['DG']['weight']['HIPP']['HIPP']]]

    # w_CA3 = [[data['connectivity']['intra']['CA3']['weight']['Pyramidal']['Pyramidal'], data['connectivity']['intra']['CA3']['weight']['Pyramidal']['Basket'], data['connectivity']['intra']['CA3']['weight']['Pyramidal']['OLM']],
    #         [data['connectivity']['intra']['CA3']['weight']['Basket']['Pyramidal'], data['connectivity']['intra']['CA3']['weight']['Basket']['Basket'], data['connectivity']['intra']['CA3']['weight']['Basket']['OLM']],
    #         [data['connectivity']['intra']['CA3']['weight']['OLM']['Pyramidal'], data['connectivity']['intra']['CA3']['weight']['OLM']['Basket'], data['connectivity']['intra']['CA3']['weight']['OLM']['OLM']]]
    
    w_CA1 = [[data['connectivity']['intra']['CA1']['weight']['Pyramidal']['Pyramidal'], data['connectivity']['intra']['CA1']['weight']['Pyramidal']['Basket'], data['connectivity']['intra']['CA1']['weight']['Pyramidal']['OLM']],
            [data['connectivity']['intra']['CA1']['weight']['Basket']['Pyramidal'], data['connectivity']['intra']['CA1']['weight']['Basket']['Basket'], data['connectivity']['intra']['CA1']['weight']['Basket']['OLM']],
            [data['connectivity']['intra']['CA1']['weight']['OLM']['Pyramidal'], data['connectivity']['intra']['CA1']['weight']['OLM']['Basket'], data['connectivity']['intra']['CA1']['weight']['OLM']['OLM']]]
    
    # w_SUB = [[data['connectivity']['intra']['Sub']['weight']['Pyramidal']['Pyramidal'], data['connectivity']['intra']['Sub']['weight']['Pyramidal']['Basket'], data['connectivity']['intra']['Sub']['weight']['Pyramidal']['OLM']],
    #         [data['connectivity']['intra']['Sub']['weight']['Basket']['Pyramidal'], data['connectivity']['intra']['Sub']['weight']['Basket']['Basket'], data['connectivity']['intra']['Sub']['weight']['Basket']['OLM']],
    #         [data['connectivity']['intra']['Sub']['weight']['OLM']['Pyramidal'], data['connectivity']['intra']['Sub']['weight']['OLM']['Basket'], data['connectivity']['intra']['Sub']['weight']['OLM']['OLM']]]
    
    # Synaptic from CA3 to CA1 area
    global syn_dist_CA3_CA1
    syn_dist_CA3_CA1 = [data['connectivity']['inter']['CA3-CA1']['syn_distance']]

    # Connection weights from CA3 to CA1 area
    global w_CA3_CA1
    w_CA3_CA1 = [data['connectivity']['inter']['CA3-CA1']['weight']]

    # synaptic properties
    global syn_threshold, syn_delay, syn_exc, syn_inh
    syn_threshold = data['synapses']['threshold']
    syn_delay = data['synapses']['delay']
    syn_exc = [data['synapses']['AMPA']['tau1'], data['synapses']['AMPA']['tau2'], data['synapses']['AMPA']['e']]
    syn_inh = [data['synapses']['GABA-A']['tau1'], data['synapses']['GABA-A']['tau2'], data['synapses']['GABA-A']['e']]

    # simulation parameters
    global duration, sim_dt, sim_v_init, sim_seed, rho, sim_celsius
    duration = data['simulation']['duration']
    sim_dt = data['simulation']['dt']
    sim_v_init = data['simulation']['v_init']
    sim_seed = data['seed_val']
    rho = data['stimulation']['rho']
    sim_celsius = data['simulation']['celsius']

    # input parameters
    global input_status, input_type, input_target, input_rate, input_amp, input_onset, input_duration
    input_status = data['input']['status']
    input_type = data['input']['type']
    input_target = data['input']['target']
    input_rate = data['input']['rate']
    input_amp = data['input']['amplitude']
    input_onset = data['input']['onset']
    input_duration = data['input']['duration']

    # extracellular stimulation parameters
    global stim_status, stim_dur, stim_amp, stim_onset, stim_pos, stim_type, stim_electrode, stim_waveform, stim_freq, stim_pulse_width, stim_interphase, ATTACHED__
    stim_status = data['stimulation']['status']
    stim_dur = data['stimulation']['duration']
    stim_amp = data['stimulation']['I']
    stim_onset = data['stimulation']['onset']
    stim_pos = data['stimulation']['coordinates']
    stim_type = data['stimulation']['type']
    stim_electrode = data['stimulation']['electrode']
    stim_waveform = data['stimulation']['waveform']
    stim_freq = data['stimulation']['frequency']
    stim_pulse_width = data['stimulation']['pulse_width']
    stim_interphase = data['stimulation']['interphase']

    ATTACHED__ = 0

    # git stuff
    global timestamp, git_branch, git_hash, git_short_hash
    timestamp = data['timestamp']
    git_branch = data['git_branch']
    git_hash = data['git_hash']
    git_short_hash = data['git_short_hash']

