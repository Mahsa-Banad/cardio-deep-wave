# Cardio Deep Wave

**Live Demo:** [cardio-deep-wave.onrender.com/docs](https://cardio-deep-wave.onrender.com/docs)
Deep Learning Pipeline for ECG Heartbeat Classification using the MIT-BIH Arrhythmia Dataset.

## Overview

Cardio Deep Wave is a machine learning project that classifies ECG heartbeats into different arrhythmia categories.

The project uses the MIT-BIH Arrhythmia Database, preprocesses ECG signals into individual heartbeat segments, and trains a neural network for heartbeat classification.

## Features

- MIT-BIH Arrhythmia Dataset support
- ECG signal loading with WFDB
- Heartbeat segmentation around R-peaks
- Label extraction from expert annotations
- Data preprocessing pipeline
- PyTorch-based neural network
- Training and validation workflow
- Performance monitoring

## Dataset

MIT-BIH Arrhythmia Database.

Dataset source:

- https://physionet.org/content/mitdb/1.0.0/

The dataset contains annotated ECG recordings with beat-level labels provided by clinical experts.

Classes:

N - Normal beat 
A - Atrial premature beat 
V - Premature ventricular contraction 

Additional classes can be added by extending the preprocessing pipeline.

---

## Project Structure

```text
cardio-deep-wave/
├── README.md
├── data
│   └── mit-bih
│       ├── (downloaded automatically)
├── main.py
├── requirements.txt
└── src
    ├── api
    │   └── server.py
    ├── data_loader
    │   ├── download_data.py
    │   └── load_ecg.py
    ├── evaluation
    │   ├── evaluate.py
    │   └── plotting.py
    ├── models
    │   └── cnn_lstm.py
    ├── preprocessing
    │   ├── feature_extraction
    │   │   └── scaterring.py
    │   ├── filtering.py
    │   ├── labels.py
    │   └── segment.py
    ├── training
    │   └── trainer.py
    └── utils
````

## Installation

Clone the repository:

```bash
git clone https://github.com/Mahsa-Banad/cardio-deep-wave.git

cd cardio-deep-wave
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate:

Mac/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---



## Run Training

```bash
python main.py
```

The pipeline will:

1. Load ECG records
2. Read annotation files
3. Extract heartbeat windows
4. Encode labels
5. Split train/validation data
6. Train the neural network
7. Report performance metrics

---

## Technologies

- Python
- NumPy
- Scikit-Learn
- PyTorch
- WFDB

---

## Future Improvements

- CNN-based heartbeat classifier
- 1D ResNet architecture
- Data augmentation
- Hyperparameter tuning
- Cross-validation
- Model deployment with FastAPI
- Docker support

---

## Results

Example training output:

```text
Epoch 1 | Loss: 0.3956 |Val Accuracy: 0.8858
Best Model Saved
Epoch 2 | Loss: 0.3477 |Val Accuracy: 0.8883
Best Model Saved
Epoch 3 | Loss: 0.3223 |Val Accuracy: 0.9003
Best Model Saved
Epoch 4 | Loss: 0.3106 |Val Accuracy: 0.9010
Best Model Saved
Epoch 5 | Loss: 0.2960 |Val Accuracy: 0.9017
Best Model Saved
Epoch 6 | Loss: 0.2853 |Val Accuracy: 0.9022
Best Model Saved
Epoch 7 | Loss: 0.2782 |Val Accuracy: 0.9026
Best Model Saved
Epoch 8 | Loss: 0.2732 |Val Accuracy: 0.9031
Best Model Saved
Epoch 9 | Loss: 0.2676 |Val Accuracy: 0.9043
Best Model Saved
Epoch 10 | Loss: 0.2636 |Val Accuracy: 0.9039
Epoch 11 | Loss: 0.2614 |Val Accuracy: 0.9057
Best Model Saved
Epoch 12 | Loss: 0.2575 |Val Accuracy: 0.9060
Best Model Saved
Epoch 13 | Loss: 0.2561 |Val Accuracy: 0.9048
Epoch 14 | Loss: 0.2557 |Val Accuracy: 0.9053
Epoch 15 | Loss: 0.2525 |Val Accuracy: 0.9017
Epoch 16 | Loss: 0.2521 |Val Accuracy: 0.9055
Epoch 17 | Loss: 0.2501 |Val Accuracy: 0.9066
Best Model Saved
Epoch 18 | Loss: 0.2479 |Val Accuracy: 0.9067
Best Model Saved
Epoch 19 | Loss: 0.2475 |Val Accuracy: 0.9071
Best Model Saved
Epoch 20 | Loss: 0.2458 |Val Accuracy: 0.9068
```

---
