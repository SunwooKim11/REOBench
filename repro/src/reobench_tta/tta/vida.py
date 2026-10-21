"""ViDA (ICLR 2024) — copied from the official repo Yangsenqiao/vida (imagenet/vida.py + imagenet/inject_vida.py,
commit c26b0db, MIT License, see LICENSE_ViDA). Changes are marked [REOBench]. Port notes: E_verify/VIDA_SPEC.md.

Kept as in the official code (it produced the paper's ImageNet-C 43.4):
  - adapters (low rank r=1, high rank r2=128) in every Linear of every `Attention` module (timm ViT: qkv, proj)
  - HKA uncertainty = 0.1 * mean variance of teacher logits over 10 CoTTA augmentations, one scalar per batch
  - set_scale bug: only scale1 (low rank) is ever set, scale2 (high rank) stays 1.0 -> fix_hka=False by default
  - symmetric CE between student and teacher; ALL parameters trained (two lr groups); EMA 0.999 / 0.8 (adapters)
  - prediction = teacher output on the clean batch, computed before the update
"""
from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
import torch.jit

from .cotta import get_tta_transforms  # [REOBench] ViDA's copy is identical to CoTTA's (shares IMG_SIZE)


# ---------------------------------------------------------------- inject_vida.py
class ViDAInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, r2 = 64):
        super().__init__()

        self.linear_vida = nn.Linear(in_features, out_features, bias)
        self.vida_down = nn.Linear(in_features, r, bias=False)
        self.vida_up = nn.Linear(r, out_features, bias=False)
        self.vida_down2 = nn.Linear(in_features, r2, bias=False)
        self.vida_up2 = nn.Linear(r2, out_features, bias=False)
        self.scale1 = 1.0
        self.scale2 = 1.0

        nn.init.normal_(self.vida_down.weight, std=1 / r**2)
        nn.init.zeros_(self.vida_up.weight)

        nn.init.normal_(self.vida_down2.weight, std=1 / r2**2)
        nn.init.zeros_(self.vida_up2.weight)

    def forward(self, input):
        return self.linear_vida(input) + self.vida_up(self.vida_down(input)) * self.scale1 + self.vida_up2(self.vida_down2(input)) * self.scale2


def inject_trainable_vida(
    model: nn.Module,
    target_replace_module: List[str] = ["CrossAttention", "Attention"],
    r: int = 4,
    r2: int = 16,
):
    """
    inject vida into model, and returns vida parameter groups.
    """

    require_grad_params = []
    names = []

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear":

                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = ViDAInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        r2,
                    )
                    _tmp.linear_vida.weight = weight
                    if bias is not None:
                        _tmp.linear_vida.bias = bias
                    _tmp.to(weight.device)  # [REOBench] official code injects on CPU before .to(device)

                    # switch the module
                    _module._modules[name] = _tmp

                    require_grad_params.extend(
                        list(_module._modules[name].vida_up.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].vida_down.parameters())
                    )
                    _module._modules[name].vida_up.weight.requires_grad = True
                    _module._modules[name].vida_down.weight.requires_grad = True

                    require_grad_params.extend(
                        list(_module._modules[name].vida_up2.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].vida_down2.parameters())
                    )
                    _module._modules[name].vida_up2.weight.requires_grad = True
                    _module._modules[name].vida_down2.weight.requires_grad = True
                    names.append(name)

    return require_grad_params, names


# ---------------------------------------------------------------- vida.py
def update_ema_variables(ema_model, model, alpha_teacher, alpha_vida):#, iteration):
    for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
        if "vida_" in name:
            ema_param.data[:] = alpha_vida * ema_param[:].data[:] + (1 - alpha_vida) * param[:].data[:]
        else:
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class ViDA(nn.Module):
    """ViDA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, ema=0.99, ema_vida = 0.99, rst_m=0.1, unc_thr = 0.2,
                 fix_hka=False, n_aug=10, unc_scale=0.1):  # [REOBench] fix_hka / n_aug / unc_scale exposed
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "vida requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()
        self.alpha_teacher = ema
        self.alpha_vida = ema_vida
        self.rst = rst_m
        self.thr = unc_thr
        self.fix_hka, self.n_aug, self.unc_scale = fix_hka, n_aug, unc_scale
        self.unc_log = []  # [REOBench] per-batch uncertainty, to check the threshold on new backbones

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # use this line if you want to reset the teacher model as well. Maybe you also
        # want to del self.model_ema first to save gpu memory.
        del self.model_ema, self.model_anchor  # [REOBench] free memory before re-copying (ViT-L)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def set_scale(self, update_model, high, low):
        for name, module in update_model.named_modules():
            if self.fix_hka and hasattr(module, 'scale1'):  # [REOBench] paper version: set both scales
                module.scale1 = low.item()
                module.scale2 = high.item()
            elif hasattr(module, 'scale1'):
                module.scale1 = low.item()
            elif hasattr(module, 'scale2'):  # never reached (official bug, kept)
                module.scale2 = high.item()

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        self.model_ema.eval()
        # Teacher Prediction
        # Augmentation-averaged Prediction
        N = self.n_aug
        outputs_uncs = []
        with torch.no_grad():  # [REOBench] outputs are detached anyway; avoids keeping 10 graphs
            for i in range(N):
                outputs_  = self.model_ema(self.transform(x)).detach()
                outputs_uncs.append(outputs_)
        outputs_unc = torch.stack(outputs_uncs)
        variance = torch.var(outputs_unc, dim=0)
        uncertainty = torch.mean(variance)*self.unc_scale
        self.unc_log.append(uncertainty.item())
        if uncertainty>= self.thr:
            lambda_high = 1+uncertainty
            lambda_low = 1-uncertainty
        else:
            lambda_low = 1+uncertainty
            lambda_high = 1-uncertainty
        self.set_scale(update_model = model, high = lambda_high, low = lambda_low)
        self.set_scale(update_model = self.model_ema, high = lambda_high, low = lambda_low)
        standard_ema = self.model_ema(x)
        outputs = self.model(x)
        # Student update
        loss = (softmax_entropy(outputs, standard_ema.detach())).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher= self.alpha_teacher, alpha_vida = self.alpha_vida)
        return standard_ema


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    vida_params_list = []
    model_params_lst = []
    for name, param in model.named_parameters():
        if 'vida_' in name:
            vida_params_list.append(param)
        else:
            model_params_lst.append(param)
    return model_params_lst, vida_params_list


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = None  # [REOBench] official deepcopy(model) is never used; skipped to save memory
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, r=1, r2=128, ckpt=None, key_map=None, allow_missing=()):
    """[REOBench] official configure_model(model, cfg): inject (ranks from cfg: vida_rank1=1, vida_rank2=128),
    optionally load the source ViDA checkpoint, model.train(). The official code wraps in DataParallel before
    loading (keys 'module.*'); here the prefix is mapped by key_map(name) -> name in `model` instead."""
    inject_trainable_vida(model=model, target_replace_module=["CrossAttention", "Attention"], r=r, r2=r2)
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu")
        sd = {key_map(k) if key_map else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)  # official: strict=True
        assert not unexpected and set(missing) <= set(allow_missing), (missing, unexpected)
    model.train()
    return model


def setup_optimizer_vida(params, param_vida, model_lr, vida_lr):
    """official imagenetc.py setup_optimizer_vida (SGD branch; cfgs/vit/vida.yaml uses SGD)."""
    return torch.optim.SGD([{"params": params, "lr": model_lr},
                            {"params": param_vida, "lr": vida_lr}],
                           lr=model_lr, momentum=0.9, dampening=0, weight_decay=0.0, nesterov=True)
