import numpy as np
from neuron import h

# input from artificial cells firing at theta frequency
def artCellsInput():
    return

# input from intracellular current with oscillatory form at specific frequency and specific amplitude
def oscInput(cell, 
             sec,
             delay: float,
             duration: float,
             freq_start: float,
             amp: float,
             freq_end: float = None):
    
    if not freq_end:
        freq_end = freq_start

    input_ = h.Izap(sec)
    input_.delay = delay
    input_.dur = duration
    input_.f0 = freq_start
    input_.f1 = freq_end
    input_.amp = amp

    cell._inputs_list.append(input_)

    return input_