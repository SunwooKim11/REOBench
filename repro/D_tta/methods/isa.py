"""Input statistics alignment (ISA) — our own training-free baseline, not from a paper.

Many RS corruptions (brightness/contrast, haze, clouds) act as global radiometric changes. ISA re-standardizes
each test batch (isa_batch) or each test image (isa_image) so that its per-channel pixel mean/std match the
clean AID training set, then runs the frozen source model. No parameters are updated.
  x' = (x - mu_test) / sigma_test * sigma_train + mu_train      (pixel space, per channel)
"""
import torch


class ISA(torch.nn.Module):
    def __init__(self, model, mean, std, level="batch"):
        super().__init__()
        self.model, self.level = model, level
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        dims = (0, 2, 3) if self.level == "batch" else (2, 3)
        mu = x.mean(dims, keepdim=True)
        sd = x.std(dims, keepdim=True).clamp_min(1e-6)
        return self.model((x - mu) / sd * self.std + self.mean)
