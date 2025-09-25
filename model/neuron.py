"""
All-in-one spiking neural network modules and functions from pytorch-spiking.
无需安装，直接调用。
"""

import torch
import numpy as np
import torch.nn as nn
import time

# 自动检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DualThresholdSelfregulatingIntegrateFunction(torch.autograd.Function):
    """
    Function for converting an arbitrary activation function to a spiking equivalent.
    """
    @staticmethod
    def forward(
        ctx,
        inputs,
        activation,
        dt=0.001,
        initial_state=None,
        spiking_aware_training=True,
        return_sequences=False,
        training=False,
    ):
        inputs = inputs.to(device)
        step = 1
        ctx.activation = activation
        ctx.return_sequences = return_sequences
        ctx.save_for_backward(inputs)
        if initial_state is None:
            initial_state = torch.rand(
                inputs.shape[0], inputs.shape[2], dtype=inputs.dtype, device=device
            )
        else:
            initial_state = initial_state.to(device)
        if training and not spiking_aware_training:
            output = activation(inputs if return_sequences else inputs[:, -1])
            return output
        inputs = inputs.to(dtype=initial_state.dtype)
        voltage = initial_state
        all_spikes = []
        rates = activation(inputs) * dt
        for i in range(inputs.shape[1]):
            for _ in range(step):
                voltage += rates[:, i]
                n_spikes = torch.floor(voltage)
                voltage -= n_spikes
            if return_sequences:
                all_spikes.append(n_spikes)
        if return_sequences:
            output = torch.stack(all_spikes, dim=1)
        else:
            output = n_spikes
        output /= dt
        return output
    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors[0]
        with torch.enable_grad():
            output = ctx.activation(inputs if ctx.return_sequences else inputs[:, -1])
            return (
                torch.autograd.grad(output, inputs, grad_outputs=grad_output)
                + (None,) * 7
            )

dual_threshold_selfregulating_integrate_autograd = DualThresholdSelfregulatingIntegrateFunction.apply

class DualThresholdSelfregulatingIntegrate(nn.Module):
    def __init__(self, activation, dt=0.001, initial_state=None, spiking_aware_training=True, return_sequences=True):
        super().__init__()
        self.activation = activation
        self.initial_state = initial_state
        self.dt = dt
        self.spiking_aware_training = spiking_aware_training
        self.return_sequences = return_sequences
    def forward(self, inputs):
        inputs = inputs.to(device)
        if self.initial_state is not None:
            initial_state = self.initial_state.to(device)
        else:
            initial_state = None
        return dual_threshold_selfregulating_integrate_autograd(
            inputs,
            self.activation,
            self.dt,
            initial_state,
            self.spiking_aware_training,
            self.return_sequences,
            self.training
        )

class Lowpass(nn.Module):
    def __init__(self, tau, units, dt=0.001, apply_during_training=True, initial_level=None, return_sequences=True):
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
        self.level_var = nn.Parameter(
            torch.zeros(1, units, device=device) if self.initial_level is None else self.initial_level.to(device)
        )
        self.smoothing_var = nn.Parameter(
            torch.ones(1, units, device=device) * self.smoothing_init
        )
    def forward(self, inputs):
        inputs = inputs.to(device)
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

class TemporalAvgPool(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, inputs):
        inputs = inputs.to(device)
        return torch.mean(inputs, dim=self.dim)

def DualThresholdSelfregulatingIntegrateAction(inputs, activation, dt=0.001, initial_state=None, return_sequences=False, use_parallel=True):
    inputs = inputs.to(device)
    batch_size, n_steps, n_neurons = inputs.shape
    if initial_state is None:
        voltage = torch.zeros(batch_size, n_neurons, dtype=inputs.dtype, device=device)
    else:
        voltage = initial_state.to(device)
    if use_parallel:
        rates = activation(inputs) * dt
        cumulative_voltage = torch.cumsum(rates, dim=1)
        spikes = torch.floor(cumulative_voltage)
        spike_diff = spikes - torch.cat([torch.zeros_like(spikes[:, :1]), spikes[:, :-1]], dim=1)
        voltage = cumulative_voltage - spikes
        if return_sequences:
            return spike_diff
        else:
            return spike_diff[:, -1]
    else:
        all_spikes = []
        rates = activation(inputs) * dt
        for i in range(inputs.shape[1]):
            voltage += rates[:, i]
            n_spikes = torch.floor(voltage)
            voltage -= n_spikes
            if return_sequences:
                all_spikes.append(n_spikes)
        if return_sequences:
            return torch.stack(all_spikes, dim=1)
        else:
            return n_spikes


            
