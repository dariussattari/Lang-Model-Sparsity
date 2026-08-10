"""Structured pruning of an ultralytics YOLOv8 detector with Torch-Pruning.

The tricky part of pruning a YOLOv8 is the C2f block: its forward does
`self.cv1(x).chunk(2, 1)`, and a channel `chunk` confuses DepGraph. The standard
fix (from Torch-Pruning's official examples/yolov8) is to swap every C2f for an
equivalent `C2f_v2` that uses two explicit 1x1 convs instead of a chunk, copying
the trained weights across. DepGraph can then trace the whole net.

After the swap we run a normal Torch-Pruning pass: GroupNormPruner +
GroupMagnitudeImportance, with the Detect head left untouched (`ignored_layers`)
so the output geometry is preserved.

C2f_v2 / transfer_weights / replace_c2f_with_c2f_v2 / infer_shortcut are taken
verbatim from Torch-Pruning/examples/yolov8/yolov8_pruning.py (MIT).
"""
import math

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics.nn.modules import Detect, C2f
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck


def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add


class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions, chunk-free so DepGraph can trace it.
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                               for _ in range(n))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels,
                            child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)


def prune_model(yolo, target_ratio, imgsz=640, iterative_steps=1, device="cpu"):
    """Structurally prune `yolo.model` in place by `target_ratio` of channels.

    Returns a stats dict (params/MACs before & after). `yolo` is mutated so the
    caller can benchmark it directly. Detect head is preserved.
    """
    model = yolo.model
    model.to(device).eval()
    # DepGraph needs a chunk-free graph.
    replace_c2f_with_c2f_v2(model)
    model.to(device)  # C2f_v2 builds fresh convs on CPU; pull them back to device

    example_inputs = torch.randn(1, 3, imgsz, imgsz).to(device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

    ignored_layers = [m for m in model.modules() if isinstance(m, Detect)]

    # per-step ratio so that after `iterative_steps` we hit target_ratio of channels
    step_ratio = 1 - math.pow((1 - target_ratio), 1 / iterative_steps)
    pruner = tp.pruner.GroupNormPruner(
        model,
        example_inputs,
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        iterative_steps=iterative_steps,
        pruning_ratio=step_ratio,
        ignored_layers=ignored_layers,
    )

    for _ in range(iterative_steps):
        pruner.step()

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)

    # keep ultralytics metadata coherent with the new tensor shapes
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.num_features = m.weight.numel()

    return {
        "target_channel_ratio": target_ratio,
        "iterative_steps": iterative_steps,
        "params_before": int(base_params),
        "params_after": int(pruned_params),
        "params_reduction_pct": round(100 * (1 - pruned_params / base_params), 1),
        "macs_before": int(base_macs),
        "macs_after": int(pruned_macs),
        "macs_reduction_pct": round(100 * (1 - pruned_macs / base_macs), 1),
    }
