"""Backbone extractor & Laplacian-based few-shot classifiers.
Implements STDA-LapShot (baseline) and STVM-LapShot (proposed).
"""
from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["get_backbone", "STDALapShot", "STVMLapShot"]

# -----------------------------------------------------------------------------
# Backbone factory -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _build_resnet(name: str = "resnet18", *, pretrained: bool = True) -> nn.Module:
    from torchvision import models

    name = name.lower()
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights as W
        weights = W.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
    elif name == "resnet34":
        from torchvision.models import ResNet34_Weights as W
        weights = W.DEFAULT if pretrained else None
        base = models.resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet variant: {name}")

    encoder = nn.Sequential(*list(base.children())[:-1])  # drop FC
    encoder.out_dim = base.fc.in_features  # type: ignore[attr-defined]

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.out_dim = encoder.out_dim  # type: ignore[attr-defined]

        def forward(self, x: torch.Tensor):  # noqa: D401, ANN201
            feats = self.encoder(x)
            return feats.flatten(1)

    return _Encoder()


def get_backbone(
    name: str = "resnet18", *, pretrained: bool = True, frozen: bool = True
) -> nn.Module:  # noqa: D401
    if name.lower().startswith("resnet"):
        net = _build_resnet(name, pretrained=pretrained)
    else:
        raise NotImplementedError(f"Backbone '{name}' not implemented.")

    if frozen:
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
    return net

# -----------------------------------------------------------------------------
# Helper routines --------------------------------------------------------------
# -----------------------------------------------------------------------------

def _self_tuning_affinity(feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = feat.size(0)
    k = max(1, int(torch.log2(torch.tensor(n, device=feat.device)).item()))
    dist = torch.cdist(feat, feat)
    _, nn_idx = torch.topk(dist, k + 1, largest=False)  # include self
    sigma = dist[torch.arange(n, device=feat.device), nn_idx[:, k]] + 1e-8

    W = torch.zeros(n, n, device=feat.device)
    for i in range(n):
        nbrs = nn_idx[i, 1:]  # drop self
        w = torch.exp(-dist[i, nbrs] / (sigma[i] * sigma[nbrs]))
        W[i, nbrs] = w
    W = (W + W.T) / 2  # symmetrise
    D_inv = 1.0 / (W.sum(1) + 1e-8)
    return W, D_inv


def _mm_iterations(
    feat: torch.Tensor,
    prototypes: torch.Tensor,
    W: torch.Tensor,
    D_inv: torch.Tensor,
    lam: torch.Tensor,
    *,
    max_iter: int = 50,
    eps: float = 1e-4,
):
    a = torch.cdist(feat, prototypes).pow(2)
    y = F.softmax(-a, dim=1)
    for _ in range(max_iter):
        y_new = F.softmax(-a + (lam * D_inv).unsqueeze(1) * (W @ y), dim=1)
        if (y_new - y).abs().max() < eps:
            break
        y = y_new
    return y

# -----------------------------------------------------------------------------
# Baseline: STDA-LapShot -------------------------------------------------------
# -----------------------------------------------------------------------------

class STDALapShot(nn.Module):
    """Self-Tuning Density-Adaptive LaplacianShot (baseline)."""

    def __init__(self):
        super().__init__()
        self.last_inference_time = 0.0

    @torch.no_grad()
    def forward(
        self, *, feat_s: torch.Tensor, y_s: torch.Tensor, feat_q: torch.Tensor
    ):  # noqa: D401, ANN201
        start = time.perf_counter()
        classes = torch.unique(y_s)
        prototypes = torch.stack([feat_s[y_s == c].mean(0) for c in classes])
        feat_all = feat_q  # no virtual nodes in baseline
        W, D_inv = _self_tuning_affinity(feat_all)
        d = torch.cdist(feat_all, prototypes).min(1).values
        lam = torch.exp(-d.pow(2) / (2 * d.median().pow(2) + 1e-8))
        y = _mm_iterations(feat_all, prototypes, W, D_inv, lam)
        preds = y.argmax(1)
        self.last_inference_time = time.perf_counter() - start
        return preds

# -----------------------------------------------------------------------------
# Proposed: STVM-LapShot -------------------------------------------------------
# -----------------------------------------------------------------------------

class STVMLapShot(STDALapShot):
    """STDA-LapShot augmented with Virtual-Mixup densification (ours)."""

    @torch.no_grad()
    def forward(
        self, *, feat_s: torch.Tensor, y_s: torch.Tensor, feat_q: torch.Tensor
    ):  # noqa: D401, ANN201
        start = time.perf_counter()
        device = feat_q.device
        classes = torch.unique(y_s)
        prototypes = torch.stack([feat_s[y_s == c].mean(0) for c in classes])
        C = len(classes)

        # Step 2 – nearest-prototype bootstrap labels -------------------------
        dist_q = torch.cdist(feat_q, prototypes)
        y_boot = dist_q.argmin(1)

        # Step 3 – label-aware virtual nodes ----------------------------------
        q_norm = F.normalize(feat_q, dim=1)
        cos = q_norm @ q_norm.T  # cosine similarity N×N
        virt = []
        for c in range(C):
            idx = (y_boot == c).nonzero(as_tuple=False).flatten()
            if idx.numel() < 2:
                continue
            cos_c = cos[idx][:, idx]
            thresh = cos_c.median()
            pairs = (cos_c >= thresh).nonzero(as_tuple=False)
            for i, j in pairs:
                if i < j:  # avoid duplicates
                    virt.append(0.5 * feat_q[idx[i]] + 0.5 * feat_q[idx[j]])
            # prototype–query mixup
            k = idx[dist_q[idx, c].argmin()]
            virt.append(0.5 * prototypes[c] + 0.5 * feat_q[k])

        if virt:
            feat_v = torch.stack(virt).to(device)
            feat_all = torch.cat([feat_q, feat_v], dim=0)
        else:
            feat_all = feat_q.clone()

        W, D_inv = _self_tuning_affinity(feat_all)
        d = torch.cdist(feat_all, prototypes).min(1).values
        lam = torch.exp(-d.pow(2) / (2 * d.median().pow(2) + 1e-8))
        y = _mm_iterations(feat_all, prototypes, W, D_inv, lam)
        preds = y[: feat_q.size(0)].argmax(1)  # discard virtual nodes for output
        self.last_inference_time = time.perf_counter() - start
        return preds
