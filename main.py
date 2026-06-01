import torch
import numpy as np
import os
import glob
from src.data_loader.download_data import download_mitdb
from src.data_loader.load_ecg import (load_record)
from src.preprocessing.filtering import banpass_filter
from src.preprocessing.segment import create_heartbeats
from src.preprocessing.labels import map_label
from src.models.cnn_lstm import CNNLSTM
from src.training.trainer import train_model
from src.evaluation.plotting import plot_curves


# device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print("Device: ", device)


#dowanload dataset
download_mitdb()


MITDB_RECORDS = [
    os.path.splitext(os.path.basename(p))[0] 
    for p in glob.glob("data/mit-bih/*.dat") 
    if os.path.exists(p.replace(".dat", ".hea"))
]

all_beats = []
all_labels = []

for record_id in MITDB_RECORDS:
    record_path = f"data/mit-bih/{record_id}"
    try:
        # load record
        signal, peaks, labels = load_record(record_path)
    except:
        print(f"[SKIP] Record {record_id}")
        continue
    #filtering
    signal = banpass_filter(signal)

    #create beats
    beats = create_heartbeats(signal, peaks)

    #labels
    mapped_labels = []
    for label in labels:
         mapped = map_label(label)
         if mapped != -1:
            mapped_labels.append(mapped)

    #align lengths
    min_len = min(
        len(beats),
          len(mapped_labels))

    beats = beats[:min_len]
    mapped_labels = mapped_labels[:min_len]

    all_beats.extend(beats)
    all_labels.extend(mapped_labels)
    print(f"[OK] Record {record_id} : {min_len} beats")
print(f"Total beats: {len(all_beats)}")


#Tensors

X = torch.tensor(
    np.array(all_beats),
    dtype = torch.float32
)

y = torch.tensor(
   all_labels,
    dtype = torch.long
)

#model
model = CNNLSTM().to(device)

#train
losses, accuracies = train_model(
    model, X, y, device
)

#plots
plot_curves(losses, accuracies)