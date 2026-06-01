from kymatio.torch import Scattering1D
import torch

def scattering_transform(signal):
    scattering =  Scattering1D(
        J=6,
        shape=signal.shape[-1],
        Q=8
    )

    signal_tensor = torch.tensor(
        signal,
        dtype=torch.float32

    )

    features = scattering(signal_tensor)
    return features

