import os
import json
import time
import subprocess
import numpy as  np

THETA  = 250 # ms
GAMMA = 25 # ms
DELSTART = 10 #  ms

# Default parameters
_data = {
    "seed_val": 42,
    "areas": {
        "EC": {
            "Pyramidal": {
                "N": 50,
                "type": "Exc",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            },
            "Basket": {
                "N": 6,
                "type": "Inh",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            },
            "OLM": {
                "N": 2,
                "type": "Inh",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            }
        },
        "DG": {
            "Granule": {
                "N": 100,
                "type": "Exc",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            },
            "Basket": {
                "N": 3,
                "type": "Inh",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            },
            "HIPP": {
                "N": 1,
                "type": "Inh",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            }
        },
        "CA3": {
            "Pyramidal": {
                "N": 26,
                "type": "Exc",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            },
            "Basket": {
                "N": 3,
                "type": "Inh",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            },
            "OLM": {
                "N": 1,
                "type": "Inh",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            }
        },
        "CA1": {
            "Pyramidal": {
                "N": 100,
                "type": "Exc",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            },
            "Basket": {
                "N": 9,
                "type": "Inh",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            },
            "OLM": {
                "N": 3,
                "type": "Inh",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            }
        },
        "Sub": {
            "Pyramidal": {
                "N": 50,
                "type": "Exc",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            },
            "Basket": {
                "N": 6,
                "type": "Inh",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            },
            "OLM": {
                "N": 2,
                "type": "Inh",
                "noise": {
                    "sigma": 1.e-2,
                    "mean": 0,
                    "tau": 25
                }
            }
        }
    },
    "inputs": {
        "CA3 schaffers": {# if network consists only of CA1
            "N_pyr": 50, # number of pyramidal cells receiving input / also total number of input cells for now (1-1 connection)
            "N_bc": 3, # number of basket cells receiving input
            "interval": THETA, # (ms) interspike interval 
            "N_spikes": 1000, # max number of spikes
            "t_start": DELSTART, # delay until first spike
            "noise": 0.0,
            "connectivity": {
                "Pyramidal": 0.03,
                "Basket": 0.025
            }
        },
        "EC perforant path": {}, # TODO
        "MS": {} # TODO
    },
    "connectivity": {
        "intra": {
            "EC": {}, # TODO
            "DG": {}, # TODO
            "CA3": {}, # TODO
            "CA1": {
                "syn_distance": { # µm
                    "Pyramidal": 500,
                    "Basket": 470,
                    "OLM": 1057.695
                },
                "weight": { # µS
                    "Pyramidal": { # from -> to
                        "Pyramidal": 0.0,
                        "Basket": 0.009, # after grid search
                        "OLM": 0.0009
                    },
                    "Basket": {
                        "Pyramidal": 0.054, # after grid search
                        "Basket": 0.0018,
                        "OLM": 0.0
                    },
                    "OLM": {
                        "Pyramidal": 0.0,
                        "Basket": 0.0,
                        "OLM": 0.0
                    }
                }
            },
            "Sub": {} # TODO
        },
        "inter": {} # TODO
    },
    "synapses": {
        "threshold": -20, # (mV)
        "delay": 1., # (ms)
        "AMPA": {
            "tau1": 0.5, # (ms) rise time
            "tau2": 3, # (ms) decay time
            "e": 0 # (mV) reversal potential
        },
        "GABA-A": {
            "tau1": 1,
            "tau2": 8,
            "e": -75
        }
    },
    "stimulation": {
        "status": False,    # False -> no stimulation; True -> stimulation is on
        "target": "CA1",
        "coordinates": (0.6105, 0.500, 0.0),  # point electrode coordinates using intrinsic coordinates system (x,y,z) (µm) 
                                                # to be transformed either to flattened coordinates or to cartesian coordinates 
        "rho": 300, # medium resistivity (ohm cm)
        "duration": 1.0, # (ms)
        "onset": 100, # (ms)
        "I": -0.75 # (mA) stimulation amplitude
    },
    "simulation": {
        "duration": 5000.0, # (ms)
        "dt": 0.025, # (ms)
        "v_init": -65, # (mV)
        "seed_noise": 0,
        "celsius": 35
    },
    "analysis": {
        "firing_rate": {
            "winsize_fr": 5, # (ms) sliding window size for computing firing rate
            "overlap_fr": 0.9, # percentage of overlap between sliding windows
        },
        "spectrogram": {
            "window_size": 1, # (ms)
            "window_size_Pxx": 1000,
            "overlap_Pxx": 0.9,
            "vmin": 1.e-12, # for color
            "vmax": 1.
        }
    },
    # git stuff
    "git_short_hash": None,
    "timestamp": None,
    "git_branch": None,
    "git_hash": None
}

def is_git_repo():
    """ Return whether current directory is a git directory """
    if subprocess.call(["git", "branch"],
            stderr=subprocess.STDOUT, stdout=open(os.devnull, 'w')) != 0:
        return False
    return True

def get_git_revision_hash():
    """ Get current git hash """
    if is_git_repo():
        answer = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'])
        return answer.decode("utf8").strip("\n")
    return "None"

def get_git_revision_short_hash():
    """ Get current git short hash """
    if is_git_repo():
        answer = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'])
        return answer.decode("utf8").strip("\n")
    return "None"

def get_git_revision_branch():
    """ Get current git branch """
    if is_git_repo():
        answer = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        return answer.decode("utf8").strip("\n")
    return "None"

def default():
    """ Get default parameters """
    _data["timestamp"] = time.ctime()
    _data["git_branch"] = get_git_revision_branch()
    _data["git_hash"] = get_git_revision_hash()
    _data["git_short_hash"] = get_git_revision_short_hash()
    return _data

def save(filename, data=None):
    """ Save parameters into a json file """
    if data is None:
       data = { name : eval(name) for name in _data.keys() if name not in ["timestamp", "git_branch", "git_hash"]}
    data["timestamp"] = time.ctime()
    data["git_branch"] = get_git_revision_branch()
    data["git_hash"] = get_git_revision_hash()
    data["git_short_hash"] = get_git_revision_short_hash()
    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4, sort_keys=False)

def load(filename):
    """ Load parameters from a json file """
    with open(filename) as infile:
        data = json.load(infile)
    return data

def dump(data):
    if not _data["timestamp"]:
        _data["timestamp"] = time.ctime()
    if not _data["git_branch"]:
        _data["git_branch"] = get_git_revision_branch()
    if not _data["git_hash"]:
        _data["git_hash"] = get_git_revision_hash()
        _data["git_short_hash"] = get_git_revision_short_hash()
    for key, value in data.items():
        print(f"{key:15s} : {value}")

if __name__  == "__main__":
    import argparse

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    if not os.path.isdir(configs_dir):
        print('[+] Creating directory', configs_dir)
        os.makedirs(configs_dir)

    parser = argparse.ArgumentParser(
        description='Generate parameters file using JSON format')
    parser.add_argument('parameters_file',
                        default='default_parameters_2',
                        type=str, nargs='?',
                        help='Parameters file (json format)')
    args = parser.parse_args()

    filename = os.path.join(configs_dir, args.parameters_file + ".json")

    print('[+] Saving parameters file {}...'.format(filename))
    save(filename, _data)
    