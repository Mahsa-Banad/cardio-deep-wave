from scipy.signal import butter, filtfilt

def banpass_filter(
        signal,
        lowcut = 0.5,
        highcut = 40,
        fs = 360,
        order = 4
):
    
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter (
        order,
        [low, high],
        btype ='band'
    )

    filtered = filtfilt(b, a, signal)
    
    return filtered
