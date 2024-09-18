import numpy as np
from scipy import signal as sig

from collections import OrderedDict


def save_membrane_potential(name_file: str, t_vec: np.ndarray, cells_potential: dict):
    # convert all values in dictionnary to np.ndarray and keys to string
    cells_potential = OrderedDict(zip([str(k) for k in cells_potential.keys()], [np.array(v) for v in cells_potential.values()]))

    # add time vector to the beginning of the ordered dictionnary
    cells_potential.update({'time': t_vec})
    cells_potential.move_to_end('time', last=False)

    np.savez(name_file, **cells_potential)


def compute_FR(spikes: np.ndarray,
               N_cells: int,
               duration: int,
               window_size: float,
               overlap: float,
               gaussian: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the firing rate using a windowed moving average.

    Parameters
    ----------
    spikes: numpy.ndarray
        The spike times (in ms)
    N_cells: int
        Population size
    duration: int
        The duration of the recording (in ms)
    window_size: float
        Width of the moving average window (in ms)
    overlap: float
        Desired overlap between the windows (percentage, in [0., 1.))
    gaussian: bool
        Apply gaussian convolution for smoothing

    Returns
    -------
    t: numpy.ndarray
        Array of time values for the computed firing rate. These are the window centers.
    FR: numpy.ndarray
        Spikes per window (needs to be normalized)
    FR_population: numpy.ndarray
        Population firing rate (Hz)

        OR
    FR_gaussian: numpy.ndarray
        Population firing rate (Hz) after gaussian convolution

        
    """

    if gaussian: # only works if every ms, TODO: improve code to accomodate for more precise time steps
        list_occ = np.zeros((duration,))
        for i in range(duration):
            list_occ[i] = len(np.where(spikes==i)[0]) 
        gaussian = np.exp(-(np.arange(-3*window_size, 3*window_size)/window_size)**2/2)

        return np.convolve(list_occ, gaussian, mode='same')
    else:
        # Calculate new sampling times
        win_step = window_size * round(1. - overlap, 4)
        # fs_n = int(1/win_step)

        # First center is at the middle of the first window
        c0 = window_size/2
        cN = duration-c0

        # centers
        centers = np.arange(c0, cN+win_step, win_step)


        # Calculate total number of spikes per window
        counts = []
        for center in centers:
            cl = center - c0
            ch = center + c0
            spike_cnt = np.count_nonzero(np.where((spikes >= cl) & (spikes < ch)))
            counts.append(spike_cnt)

        # return centers, spike counts per window and population firing rate
        return centers, np.array(counts), np.array(counts)/(window_size*1e-3)/N_cells
    

def my_specgram(signal: np.ndarray,
                   fs: int,
                   window_width: int,
                   window_overlap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the power spectrum of the specified signal.

    A periodic Hann window with the specified width and overlap is used.

    Parameters
    ----------
    signal: numpy.ndarray
        The input signal
    fs: int
        Sampling frequency of the input signal
    window_width: int
        Width of the Hann windows in samples
    window_overlap: int
        Overlap between Hann windows in samples

    Returns
    -------
    f: numpy.ndarray
        Array of frequency values for the first axis of the returned spectrogram
    t: numpy.ndarray
        Array of time values for the second axis of the returned spectrogram
    sxx: numpy.ndarray
        Power spectrogram of the input signal with axes [frequency, time]
    """
    f, t, Sxx = sig.spectrogram(x=signal,
                                # nfft=2048,
                                detrend=False,
                                fs=fs,
                                window=sig.windows.hann(M=window_width, sym=False),
                                nperseg=window_width,
                                noverlap=window_overlap,
                                return_onesided=True,
                                # scaling='spectrum',
                                mode='magnitude')

    return f, t, (1.0 / window_width) * (Sxx ** 2)

