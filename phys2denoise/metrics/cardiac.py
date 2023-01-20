"""Denoising metrics for cardio recordings."""
import numpy as np

from .. import references
from ..due import due


def iht():
    """Calculate instantaneous heart rate."""
    pass


@due.dcite(references.CHANG_GLOVER_2009)
def crf(samplerate, oversampling=50, time_length=32, onset=0.0, tr=2.0):
    """Calculate the cardiac response function using Chang and Glover's definition.

    Parameters
    ----------
    samplerate : :obj:`float`
        Sampling rate of data, in seconds.
    oversampling : :obj:`int`, optional
        Temporal oversampling factor, in seconds. Default is 50.
    time_length : :obj:`int`, optional
        RRF kernel length, in seconds. Default is 32.
    onset : :obj:`float`, optional
        Onset of the response, in seconds. Default is 0.

    Returns
    -------
    crf : array-like
        Cardiac or "heart" response function

    Notes
    -----
    This cardiac response function was defined in [1]_, Appendix A.

    The core code for this function comes from metco2, while several of the
    parameters, including oversampling, time_length, and onset, are modeled on
    nistats' HRF functions.

    References
    ----------
    .. [1] C. Chang & G. H. Glover, "Relationship between respiration,
       end-tidal CO2, and BOLD signals in resting-state fMRI," Neuroimage,
       issue 47, vol. 4, pp. 1381-1393, 2009.
    """

    def _crf(t):
        rf = 0.6 * t ** 2.7 * np.exp(-t / 1.6) - 16 * (
            1 / np.sqrt(2 * np.pi * 9)
        ) * np.exp(-0.5 * (((t - 12) ** 2) / 9))
        return rf

    dt = tr / oversampling
    time_stamps = np.linspace(
        0, time_length, np.rint(float(time_length) / dt).astype(np.int)
    )
    time_stamps -= onset
    crf_arr = _crf(time_stamps)
    crf_arr = crf_arr / max(abs(crf_arr))
    return crf_arr

@due.dcite(references.CHEN_LEWIS_2020)
def hbi(card, samplerate, window=6):
    """Calculate the median heart beats interval (HBI) in a sliding window.

    Parameters
    ----------
    card : (X,) :obj:`Physio`
        A Physio object with the following parameters: the heart peaks index, card.peaks, and card.data, the ppg values.
    samplerate : :obj:`float`
        Sampling rate for card, in Hertz.
    window : :obj:`int`, optional
        Size of the sliding window, in seconds.
        By default at 6.

    Returns
    -------
    hbi_out : (X, 2) :obj:`numpy.ndarray`
        Median of heart beats interval values.
        The first column is raw HBI values, after normalization.
        The second column is HBI values convolved with the RRF, after normalization.

    Notes
    -----
    Heart beats interval (HBI) was introduced in [1]_, and consists of the
    average of the time interval between two heart beats based on ppg data within a 6-second window.

    This metric is often lagged back and/or forward in time and convolved with
    an inverse of the cardiac response function before being included in a GLM.
    Regressors also often have mean and linear trends removed and are
    standardized prior to regressions.

    References
    ----------
    .. [1] J. E. Chen & L. D. Lewis, "Resting-state "physiological networks"", Neuroimage, 
        vol. 213, pp. 116707, 2020.
    
      
    """
   
    # Convert window to time points
    size = card.data.shape[0] 
    window_tp = int(window * samplerate) 
    hbi_arr = np.empty(size)
    
    for i in range(size):
        if i==0 or i==(size-1):
            print(i)
            window_tp = 0
    
        elif i<120:
            print(i)
            window_tp = i * 2 * samplerate 
    
        elif i>(size-1-120):
            print(i)
            window_tp = (size - 1 - i) * 2 * samplerate
            
        else:
            window_tp = window
        
        peaks = card.peaks[(card.peaks >= (i-window_tp/2)) & (card.peaks <= (i+window_tp/2))]
        hbi_arr[i] = np.ediff1d(peaks).median()

    # Convolve with crf
    crf_arr = crf(samplerate, oversampling=50)
    icrf_arr = - crf_arr
    hbi_convolved = convolve1d(hbi_arr, icrf_arr, axis=0)

    # Concatenate the raw and convolved versions
    hbi_combined = np.stack((hbi_arr, hbi_convolved), axis=-1)

    # Detrend and normalize
    hbi_combined = hbi_combined - np.mean(hbi_combined, axis=0)
    hbi_combined = detrend(hbi_combined, axis=0)
    hbi_out = zscore(hbi_combined, axis=0)
    return hbi_out



def cardiac_phase(peaks, sample_rate, slice_timings, n_scans, t_r):
    """Calculate cardiac phase from cardiac peaks.

    Assumes that timing of cardiac events are given in same units
    as slice timings, for example seconds.

    Parameters
    ----------
    peaks : 1D array_like
        Cardiac peak times, in seconds.
    sample_rate : float
        Sample rate of physio, in Hertz.
    slice_timings : 1D array_like
        Slice times, in seconds.
    n_scans : int
        Number of volumes in the imaging run.
    t_r : float
        Sampling rate of the imaging run, in seconds.

    Returns
    -------
    phase_card : array_like
        Cardiac phase signal, of shape (n_scans,)
    """
    assert slice_timings.ndim == 1, "Slice times must be a 1D array"
    n_slices = np.size(slice_timings)
    phase_card = np.zeros((n_scans, n_slices))

    card_peaks_sec = peaks / sample_rate
    for i_slice in range(n_slices):
        # generate slice acquisition timings across all scans
        times_crSlice = t_r * np.arange(n_scans) + slice_timings[i_slice]
        phase_card_crSlice = np.zeros(n_scans)
        for j_scan in range(n_scans):
            previous_card_peaks = np.asarray(
                np.nonzero(card_peaks_sec < times_crSlice[j_scan])
            )
            if np.size(previous_card_peaks) == 0:
                t1 = 0
            else:
                last_peak = previous_card_peaks[0][-1]
                t1 = card_peaks_sec[last_peak]
            next_card_peaks = np.asarray(
                np.nonzero(card_peaks_sec > times_crSlice[j_scan])
            )
            if np.size(next_card_peaks) == 0:
                t2 = n_scans * t_r
            else:
                next_peak = next_card_peaks[0][0]
                t2 = card_peaks_sec[next_peak]
            phase_card_crSlice[j_scan] = (
                2 * np.math.pi * (times_crSlice[j_scan] - t1)
            ) / (t2 - t1)
        phase_card[:, i_slice] = phase_card_crSlice

    return phase_card
