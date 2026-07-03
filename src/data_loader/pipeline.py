import os
import glob



from src.data_loader.load_ecg import load_record
from src.preprocessing.filtering import bandpass_filter
from src.preprocessing.segment import create_heartbeats
from src.preprocessing.labels import map_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "saved_models", "best_model.pth")


def get_available_records(data_dir ="data/mit-bih"):
    return{
        os.path.splitext(os.path.basename(p))[0]
         for p in glob.glob("data/mit-bih/*.dat")
        if os.path.exists(p.replace(".dat", ".hea"))
    }
def process_record(record_id, data_dir ="data/mit-bih"):
    record_path = f"{data_dir}/{record_id}"

    signal, peaks, labels = load_record(record_path)
    signal = bandpass_filter(signal)
    beats = create_heartbeats(signal, peaks)

    mapped_labels = []
    for label in labels:
        mapped = map_label(label)
        if mapped!= -1:
            mapped_labels.append(mapped)
    
    min_len = min(len(beats), len(mapped_labels))
    beats = beats[:min_len]
    mapped_labels = mapped_labels[:min_len]
    
    if min_len == 0:
        return{"error": f"No valid beats found in record {record_id}"}
    
    return beats, mapped_labels