#!/usr/bin/env python3
"""
统计模型参数量与计算量（单位：参数 M，FLOPs G）。

用法示例：
    python analysis/model_stats.py --model_type textcnn --model_mode snn

脚本说明：
 - 优先尝试使用 thop.profile 计算 FLOPs；若 thop 不可用或在自定义模块上失败，
   会退回到仅统计 nn.Conv2d 与 nn.Linear 的理论 FLOPs（乘加计为 2 次浮点运算）。
 - 参数统计使用 model.parameters() 的精确参数个数。
"""
import argparse
import importlib
import sys
from typing import Tuple
import pathlib
import importlib.util
import os

# Ensure the project root (snn-main) is on sys.path so `import model.*` works
_THIS_FILE = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn


def build_args_from_defaults():
    # load SNNArgs defaults if available
    try:
        from args import SNNArgs
        args = SNNArgs()
        args.renew_args()
        return args
    except Exception:
        # minimal fallback
        class _A:
            sentence_length = 25
            hidden_dim = 300
            filters = [3, 4, 5]
            filter_num = 100
            label_num = 2
            dropout_p = 0.5
            beta = 1.0
            threshold = 1.0
        return _A()


def create_model(args_obj, model_type: str, model_mode: str):
    """根据 model_type 与 model_mode 构造模型实例。
    支持：
      - model_type textcnn: model/textcnn.SNN_TextCNN (model_mode='snn')
      - model_type textcnn: model/normal_textcnn.Normal_TextCNN (model_mode='normal' 或 'ann')
      - model_type ann: model/ann_model.ANN_TextCNN (model_mode='ann')
    """
    model_type = model_type.lower()
    model_mode = model_mode.lower()

    def safe_import_model_module(mod_name: str):
        """Try normal package import first; if that fails, load the module file directly to avoid executing model.__init__.py."""
        full_name = f'model.{mod_name}'
        try:
            return importlib.import_module(full_name)
        except Exception:
            # fallback: load from file path model/<mod_name>.py
            module_path = os.path.join(str(_PROJECT_ROOT), 'model', f'{mod_name}.py')
            if not os.path.exists(module_path):
                raise
            spec = importlib.util.spec_from_file_location(full_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    if model_type == 'textcnn':
        if model_mode == 'snn':
            mmod = safe_import_model_module('textcnn')
            Model = getattr(mmod, 'SNN_TextCNN')
            return Model(args_obj)
        elif model_mode == 'ann':
            mmod = safe_import_model_module('ann_model')
            Model = getattr(mmod, 'ANN_TextCNN')
            return Model(args_obj)
        else:
            # default -> normal
            mmod = safe_import_model_module('normal_textcnn')
            Model = getattr(mmod, 'Normal_TextCNN')
            return Model(args_obj)

    # try to import a model module with same name
    try:
        mmod = safe_import_model_module(model_type)
        # try common class names
        for cls in ['Model', 'Net', 'Network']:
            if hasattr(mmod, cls):
                return getattr(mmod, cls)(args_obj)
        # fallback: find first nn.Module subclass in module
        for name in dir(mmod):
            obj = getattr(mmod, name)
            try:
                if isinstance(obj, type) and issubclass(obj, nn.Module):
                    return obj(args_obj)
            except Exception:
                continue
    except Exception:
        pass

    raise RuntimeError(f'Cannot construct model for type={model_type}, mode={model_mode}')


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def flops_with_thop(model: nn.Module, input_tensor: torch.Tensor) -> Tuple[float, int]:
    try:
        from thop import profile
    except Exception:
        raise
    model.eval()
    with torch.no_grad():
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops, params


def flops_by_hooks(model: nn.Module, input_tensor: torch.Tensor, time_steps: int = 1) -> float:
    """使用 forward hook 统计 Conv2d 与 Linear 的理论 FLOPs（乘加记为 2）。
    额外：对 snntorch 的 Leaky 等神经元模块做简单估算（每个神经元约 4 次浮点运算/步），
    并根据 time_steps 乘上时间步数以估计时序展开的计算量。
    """

    hooks = []
    hooks_flops = []

    def conv_hook(self, input, output):
        # output: N x Cout x Hout x Wout
        out = output[0] if isinstance(output, (list, tuple)) else output
        if out is None:
            return
        batch_size, Cout, Hout, Wout = out.shape
        Cin = self.in_channels
        Kh, Kw = self.kernel_size
        groups = getattr(self, 'groups', 1)
        # per output position, conv does Kh*Kw*Cin/groups * Cout multiplications
        ops_per_position = Kh * Kw * (Cin // groups) * Cout
        total_ops = ops_per_position * Hout * Wout * batch_size
        # multiply-add counts as 2 FLOPs
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

    def leaky_hook(self, input, output):
        """Estimate operations for snntorch Leaky/LIF-like modules.
        Conservative estimate per element per time step:
          - mem decay (mul): 1
          - mem add input (add): 1
          - compare to threshold (comp): 1
          - reset/subtract (add/sub): 1
        = 4 FLOPs per element per time step (multiply-add counted separately where relevant).
        """
        out = output[0] if isinstance(output, (list, tuple)) else output
        if out is None:
            return
        # number of elements processed (batch * features * any spatial dims)
        num_elems = out.numel()
        # per element ops
        ops_per_elem = 4
        hooks_flops.append(num_elems * ops_per_elem)

    for m in model.modules():
        # Conv2d and Linear: measured precisely
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        else:
            # try to detect snntorch Leaky/LIF modules by class name to avoid importing snntorch types
            cls_name = m.__class__.__name__
            module_path = m.__class__.__module__
            if 'snntorch' in module_path and cls_name.lower() in ('leaky', 'lif'):
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
    # multiply by time steps (for SNN temporal unfolding)
    if time_steps is None:
        time_steps = 1
    flops = flops * int(time_steps)
    return flops


def main():
    parser = argparse.ArgumentParser(description='统计模型参数与 FLOPs')
    parser.add_argument('--model_type', type=str, default='textcnn', help='模型类型（例如 textcnn）')
    parser.add_argument('--model_mode', type=str, default='snn', help='模型模式，snn/ann/normal')
    parser.add_argument('--batch', type=int, default=1, help='用于计算 FLOPs 的 batch 大小')
    parser.add_argument('--use_thop', action='store_true', help='强制使用 thop（若不可用会报错）')
    parser.add_argument('--time_steps', type=int, default=None, help='SNN 时间步数；默认使用 args.num_steps 或 1')
    parser.add_argument('--bert_input', action='store_true', help='使用 BERT base 的输入维度 hidden_dim=768（保持与 Bert-base 一致）')
    parser.add_argument('--include_bert', action='store_true', help='将完整的 BERT-base 模型（参数量约 110M）并入统计（需要 transformers）')
    parser.add_argument('--show_structure', action='store_true', help='打印模型网络结构（逐层模块及其参数数量）')
    args_cli = parser.parse_args()

    args_obj = build_args_from_defaults()

    # override args defaults with CLI options if provided
    if args_cli.bert_input:
        # BERT-base hidden size
        setattr(args_obj, 'hidden_dim', 768)
    # allow user to override some common dimensions via env/CLI in future; keep for now

    print(f'构建模型 type={args_cli.model_type} mode={args_cli.model_mode} ...')
    model = create_model(args_obj, args_cli.model_type, args_cli.model_mode)
    model.eval()

    def print_model_structure(m: nn.Module):
        """Print a readable module list with parameter counts."""
        print('--- 模型结构（逐层） ---')
        print(m)
        print('--- 逐层参数统计 ---')
        total = 0
        trainable = 0
        for name, module in m.named_modules():
            # skip the top-level module printed above
            if name == '':
                continue
            p_total = sum(p.numel() for p in module.parameters())
            p_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total += p_total
            trainable += p_train
            print(f'{name:40s} | {module.__class__.__name__:20s} | params: {p_total:10d} | trainable: {p_train:10d}')
        print('--- 总计 ---')
        print(f'module params sum (may double-count nested): {total}    trainable sum: {trainable}')
        print('注意：上面统计为每个子模块的参数数目，顶层参数总量以最终报告为准（避免重复统计）。')

    if args_cli.show_structure:
        print_model_structure(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # input shape: (batch, sentence_length, hidden_dim)
    batch = args_cli.batch
    inp = torch.randn(batch, args_obj.sentence_length, args_obj.hidden_dim, device=device)

    # Try thop first unless user disabled it
    flops = None
    thop_params = None
    if not args_cli.use_thop:
        try:
            from thop import profile  # type: ignore
            try:
                flops, thop_params = flops_with_thop(model, inp)
            except Exception as e:
                # thop import ok but profile failed on custom modules; fallback
                flops = None
        except Exception:
            flops = None

    if args_cli.use_thop:
        try:
            flops, thop_params = flops_with_thop(model, inp)
        except Exception as e:
            print('thop 计算 FLOPs 失败：', e)
            sys.exit(1)

    if flops is None:
        print('使用回退方法（统计 Conv2d/Linear 的理论 FLOPs，并估算 snntorch 神经元）')
        time_steps = args_cli.time_steps
        if time_steps is None:
            # try to read from args_obj
            time_steps = getattr(args_obj, 'num_steps', None) or getattr(args_obj, 'num_steps', None) or getattr(args_obj, 'num_steps', 1)
        flops = flops_by_hooks(model, inp, time_steps=time_steps)

    # Count parameters (model only)
    params = count_parameters(model)

    # Optionally include BERT-base model parameters
    bert_params = 0
    bert_flops = 0
    if args_cli.include_bert:
        try:
            from transformers import AutoModel
            print('加载 BERT-base 模型以并入统计（稍慢）...')
            bert = AutoModel.from_pretrained('bert-base-uncased')
            bert.eval()
            bert_params = sum(p.numel() for p in bert.parameters())
            # try to get FLOPs for bert if thop available
            try:
                from thop import profile
                # create sample inputs for BERT: input_ids and attention_mask
                seq_len = getattr(args_obj, 'sentence_length', 128)
                input_ids = torch.ones(1, seq_len, dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids, device=device)
                # Some HF models expect dict inputs; profile may not support it. We'll try measuring embedding+encoder only via bert.forward
                with torch.no_grad():
                    bert_flops, _ = profile(bert, inputs=(input_ids, attention_mask), verbose=False)
            except Exception:
                bert_flops = 0
        except Exception as e:
            print('无法加载 transformers 或 BERT 模型，跳过 BERT 参数/FLOPs 的细粒度统计：', e)
            # fallback: assume ~110M params
            bert_params = 110_000_000
            bert_flops = 0

    # sum params and flops
    total_params = params + bert_params
    total_flops = flops + bert_flops

    params_m = total_params / 1e6
    flops_g = total_flops / 1e9

    print('模型（主干）：', model.__class__.__name__)
    if args_cli.include_bert:
        print('并入 BERT-base（encoder）参数与 FLOPs：')
        print(f'  BERT 参数: {bert_params} ({bert_params/1e6:.4f} M)')
        if bert_flops:
            print(f'  BERT FLOPs (approx): {bert_flops:.0f} ({bert_flops/1e9:.4f} G)')
        else:
            print('  BERT FLOPs: 未能计算（thop/transformers 支持或失败），显示参数量仅作为参考')

    print(f'总参数量 (包含可选 BERT): {total_params} ({params_m:.4f} M)')
    print(f'总 FLOPs (approx): {total_flops:.0f} ({flops_g:.4f} G)')


if __name__ == '__main__':
    main()
