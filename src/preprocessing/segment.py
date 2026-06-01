import numpy as np 

def create_heartbeats(
        signal,
        peaks,
        window = 180
):
    
    beats = []
    
    for peak in peaks:
        start = peak - window
        end = peak + window

        if start < 0 or end >= len(signal):
            continue
        beat = signal[start : end]
        beats.append(beat)

    return np.array(beats)