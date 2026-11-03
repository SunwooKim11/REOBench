"""LSCD-TTA (Liang et al., JAG 2025) — loss ported verbatim from the official code
(github.com/AEAQA/LSCD-TTA, models/cmp_losses.py `lscd`). The released adaptation loop does not run
(SyntaxError in models/AdaptorHelper.py:116), so the online loop follows Tent: one forward on the batch,
one update, return the pre-update logits (paper Sec. 2.1; the code instead re-forwards after the update).
Updated params: all LayerNorm affine params (AdaptorHelper falls back from 'bn' to all nn.LayerNorm for ViT).
"""
from copy import deepcopy

import torch
import torch.nn as nn


def lscd_loss(logits, div=True):  # = cmp_losses.py lscd(), unused wcce term dropped
    # div=False: paper Eq. 13 (alpha WCSE + beta BCSE + tau LSD); the SHOT-style diversity term is code-only
    label_smoothing_args = 1e-2
    prob = logits.softmax(dim=1)
    pseudo_inds = prob.argmax(dim=1)

    num_classes = logits.size(1)
    soft_one_hot = torch.zeros_like(prob)
    soft_one_hot.fill_(label_smoothing_args / (num_classes - 1))
    soft_one_hot.scatter_(1, pseudo_inds.unsqueeze(1), 1.0 - label_smoothing_args)
    target_one_hot = torch.zeros_like(prob).scatter_(1, pseudo_inds.unsqueeze(1), 1)

    balance_soft_one_hot = (1 - soft_one_hot) * (1 - prob) + soft_one_hot * prob

    # ncse
    beta = 1
    neg_inds = target_one_hot.lt(1).float()
    neg_weights = torch.pow(1 - soft_one_hot, beta)
    neg_conf_prob = neg_inds * neg_weights
    ncse_loss = - (torch.exp(neg_conf_prob) * prob**0.5 * (prob + 1e-8).log()).sum(dim=1).mean()

    # bse
    bse_loss = - (torch.exp((balance_soft_one_hot)) * prob * (prob + 1e-8).log()).sum(dim=1).mean()

    # div (for stable)
    msoftmax = prob.mean(dim=0)
    diversity_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-8))

    # lsd
    lsd_arg = 1e-3      # alpha
    div_arg = 4     # beta
    lsd_loss = -(prob * (1 / (1 - prob + lsd_arg)).log()).sum(dim=1).mean()
    merge_lsd_loss = lsd_loss + (div_arg * diversity_loss if div else 0)

    return 0.25 * ncse_loss + 1.0 * bse_loss + 1.5 * merge_lsd_loss


class LSCD(nn.Module):
    def __init__(self, model, optimizer, div=True):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.div = div

    @torch.enable_grad()
    def forward(self, x):
        outputs = self.model(x)
        loss = lscd_loss(outputs, self.div)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
    return model


def collect_params(model):
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            for np_, p in m.named_parameters():
                if np_ in ["weight", "bias"]:
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names
