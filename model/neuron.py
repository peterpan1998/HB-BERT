import numpy as np
import torch
import torch.nn as nn
class DualThresholdSelfregulatingIntegrateFunction(torch.autograd.Function):
    #There is a problem with the previous one, which will be uploaded after modification
    pass
class DualThresholdSelfregulatingIntegrate(torch.nn.Module):  # pylint: disable=abstract-method
    def __init__(self,activation,dt=0.001,initial_state=None,spiking_aware_training=True,return_sequences=True,T=1):
        super().__init__()
        # 定义可学习电压阈值参数
        #self.voltage_threshold = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.activation = activation
        self.initial_state = initial_state
        self.dt = dt
        self.T = T
        self.spiking_aware_training = spiking_aware_training
        self.return_sequences = return_sequences
    def forward(self, inputs):
        return DualThresholdSelfregulatingIntegrateFunction.apply(
            inputs,
            self.activation,
            self.dt,
            self.initial_state,
            self.spiking_aware_training,
            self.return_sequences,
            self.training,
            self.T
        )
class Lowpass(torch.nn.Module):  # pylint: disable=abstract-method
    def __init__(self,tau,units,dt=0.001,apply_during_training=True,initial_level=None,return_sequences=True,):
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be a positive number")
        self.tau = tau
        self.units = units
        self.dt = dt
        self.apply_during_training = apply_during_training
        self.initial_level = initial_level
        self.return_sequences = return_sequences
        smoothing_init = np.exp(-self.dt / self.tau)
        self.smoothing_init = np.log(smoothing_init / (1 - smoothing_init))
        self.level_var = torch.nn.Parameter(
            torch.zeros(1, units) if self.initial_level is None else self.initial_level
        )
        self.smoothing_var = torch.nn.Parameter(
            torch.ones(1, units) * self.smoothing_init
        )
    def forward(self, inputs):
        if self.training and not self.apply_during_training:
            return inputs if self.return_sequences else inputs[:, -1]
        level = self.level_var
        smoothing = torch.sigmoid(self.smoothing_var)
        inputs = inputs.type(self.smoothing_var.dtype)
        all_levels = []
        for i in range(inputs.shape[1]):
            level = (1 - smoothing) * inputs[:, i] + smoothing * level
            if self.return_sequences:
                all_levels.append(level)
        if self.return_sequences:
            return torch.stack(all_levels, dim=1)
        else:
            return level
class TemporalAvgPool(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, inputs):
        return torch.mean(inputs, dim=self.dim)
__all__ = [
    "DualThresholdSelfregulatingIntegrate",
    "Lowpass",
    "TemporalAvgPool",
]
