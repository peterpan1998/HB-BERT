#!/usr/bin/env python3
"""
Standalone script to build SNN model like in main.py and report parameter count and FLOPs.

Usage examples:
  python analysis/snn_stats_from_main.py --model_type textcnn --model_mode snn --bert_input --time_steps 50 --show_structure

This script avoids importing `main.py` to prevent executing package-level imports that may fail.
It loads the model implementation directly from `model/<name>.py`.
"""
import argparse
import pathlib
import sys
import importlib.util
import os
from typing import Tuple

# Ensure project root on path
_THIS = pathlib.Path(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn


def build_args_defaults():
    try:
        from args import SNNArgs
        a = SNNArgs()
        a.renew_args()
        return a
    except Exception:
        class _A:
            sentence_length = 25
            hidden_dim = 300
            filters = [3, 4, 5]
            filter_num = 100
            label_num = 2
            dropout_p = 0.5
            beta = 1.0
            threshold = 1.0
            num_steps = 50
        return _A()


def safe_load_module(module_name: str, rel_path: str):
    module_path = os.path.join(str(_ROOT), rel_path)
    if not os.path.exists(module_path):
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def flops_by_hooks(model: nn.Module, input_tensor: torch.Tensor, time_steps: int = 1) -> float:
    hooks = []
    hooks_flops = []

    def conv_hook(self, input, output):
        out = output[0] if isinstance(output, (list, tuple)) else output
        if out is None:
            return
        batch_size, Cout, Hout, Wout = out.shape
        Cin = self.in_channels
        Kh, Kw = self.kernel_size
        groups = getattr(self, 'groups', 1)
        ops_per_position = Kh * Kw * (Cin // groups) * Cout
        total_ops = ops_per_position * Hout * Wout * batch_size
        hooks_flops.append(total_ops * 2)

    def linear_hook(self, input, output):
        out = output[0] if isinstance(output, (list, tuple)) else output
        if out is None:
            return
        batch_size = out.shape[0]
        in_features = self.in_features
        out_features = self.out_features
        total_ops = batch_size * in_features * out_features
        hooks_flops.append(total_ops * 2)

    def leaky_hook(input_self, input, output):
        out = output[0] if isinstance(output, (list, tuple)) else output
        if out is None:
            return
        num_elems = out.numel()
        ops_per_elem = 4
        hooks_flops.append(num_elems * ops_per_elem)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        else:
            cls_name = m.__class__.__name__.lower()
            module_path = m.__class__.__module__
            if 'snntorch' in module_path and cls_name in ('leaky', 'lif'):
                hooks.append(m.register_forward_hook(leaky_hook))

    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    flops = sum(hooks_flops)
    flops = flops * max(1, int(time_steps))
    return flops


def print_structure(model: nn.Module):
    print('--- model ---')
    print(model)
    print('--- modules param counts ---')
    for name, module in model.named_modules():
        if name == '':
            continue
        p = sum(p.numel() for p in module.parameters())
        pt = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f'{name:40s} | {module.__class__.__name__:20s} | params: {p:10d} | trainable: {pt:10d}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='textcnn')
    parser.add_argument('--model_mode', default='snn')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--time_steps', type=int, default=None)
    parser.add_argument('--bert_input', action='store_true')
    parser.add_argument('--show_structure', action='store_true')
    parser.add_argument('--use_thop', action='store_true')
    args = parser.parse_args()

    cfg = build_args_defaults()
    # optional overrides to match main.py behaviour
    if args.bert_input:
        setattr(cfg, 'hidden_dim', 768)
    cfg.model_type = args.model_type
    cfg.model_mode = args.model_mode

    # build surrogate similar to main.build_surrogate
    try:
        import snntorch.surrogate as surrogate
        if getattr(cfg, 'surrogate', 'fast_sigmoid') == 'fast_sigmoid':
            spike_grad = surrogate.fast_sigmoid()
        else:
            spike_grad = surrogate.fast_sigmoid()
    except Exception:
        spike_grad = None

    # load model implementation file directly
    if cfg.model_type == 'textcnn' and cfg.model_mode == 'snn':
        mod = safe_load_module('model.textcnn', os.path.join('model', 'textcnn.py'))
        ModelClass = getattr(mod, 'SNN_TextCNN')
        model = ModelClass(cfg, spike_grad=spike_grad)
        # call initial if present
        if hasattr(model, 'initial'):
            try:
                model.initial()
            except Exception:
                pass
    else:
        raise NotImplementedError('This helper currently supports only snn textcnn')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    batch = args.batch
    inp = torch.randn(batch, cfg.sentence_length, cfg.hidden_dim, device=device)

    flops = None
    if args.use_thop:
        try:
            from thop import profile
            with torch.no_grad():
                flops, _ = profile(model, inputs=(inp,), verbose=False)
        except Exception as e:
            print('thop failed, falling back to hooks:', e)
            flops = None

    if flops is None:
        time_steps = args.time_steps if args.time_steps is not None else getattr(cfg, 'num_steps', 1)
        flops = flops_by_hooks(model, inp, time_steps=time_steps)

    params = count_parameters(model)
    print('Model:', model.__class__.__name__)
    print(f'Parameters: {params} ({params/1e6:.4f} M)')
    print(f'FLOPs (approx): {flops:.0f} ({flops/1e9:.4f} G)')

    if args.show_structure:
        print_structure(model)


if __name__ == '__main__':
    main()
#python3 analysis/snn_model_p_g.py --model_type textcnn --model_mode snn --bert_input --time_steps 50 --show_structure