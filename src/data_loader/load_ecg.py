import wfdb

def load_record(record_path):
    record = wfdb.rdrecord(record_path)

    annotation = wfdb.rdann(
        record_path, 'atr'
    )

    signal = record.p_signal[:, 0]
    labels = annotation.symbol
    peaks = annotation.sample

    return signal, peaks, labels