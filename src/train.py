"""Single experiment executor.
Trains / evaluates one run defined in config/runs/*.yaml using Hydra.
Integrates full WandB logging and performs all mandatory assertions.
"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import List

# -----------------------------------------------------------------------------
# Cache directories (must be set *before* torch / HF are imported) -------------
# -----------------------------------------------------------------------------
os.environ.setdefault("TORCH_HOME", ".cache/torch")
os.environ.setdefault("HF_HOME", ".cache/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", ".cache/transformers")

# -----------------------------------------------------------------------------
# Heavy imports ----------------------------------------------------------------
# -----------------------------------------------------------------------------
import numpy as np  # noqa: E402  pylint: disable=C0413
import torch  # noqa: E402  pylint: disable=C0413
import torch.nn.functional as F  # noqa: E402  pylint: disable=C0413
import hydra  # noqa: E402  pylint: disable=C0413
from omegaconf import DictConfig, OmegaConf  # noqa: E402  pylint: disable=C0413
from tqdm import tqdm  # noqa: E402  pylint: disable=C0413
import wandb  # noqa: E402  pylint: disable=C0413

from src import model as model_lib  # noqa: E402  pylint: disable=C0413
from src import preprocess  # noqa: E402  pylint: disable=C0413

# -----------------------------------------------------------------------------
# Utility helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------

def _set_seeds(seed: int = 2026) -> None:  # noqa: D401
    """Establish reproducible behaviour (within reason)."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # keep fast kernels
    torch.backends.cudnn.benchmark = True


def _wald_ci95(num_correct: int, num_total: int, z: float = 1.96) -> float:
    if num_total == 0:
        return 0.0
    p_hat = num_correct / num_total
    return float(z * math.sqrt(p_hat * (1.0 - p_hat) / num_total))


# -----------------------------------------------------------------------------
# Hydra entrypoint -------------------------------------------------------------
# -----------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../config", config_name="config")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:  # noqa: D401
    """Main routine for a *single* configured run."""

    # ------------------------- Sanity checks ---------------------------------
    if not hasattr(cfg, "run") or cfg.run is None:
        raise ValueError("Missing run configuration: specify run=<run_id> on CLI.")
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be one of ['trial','full']")

    # Apply trial-mode light-weight overrides ---------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.dataset.n_episodes = min(int(cfg.dataset.n_episodes), 2)
        if hasattr(cfg.training, "epochs") and cfg.training.epochs:
            cfg.training.epochs = 1

    # ------------------------- Reproducibility & device ----------------------
    _set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------- WandB initialisation --------------------------
    wandb_run = None
    if cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode,
        )
        print("[W&B]", wandb_run.url, flush=True)

    # ------------------------- Backbone & classifier -------------------------
    backbone = model_lib.get_backbone(
        name=cfg.model.name,
        pretrained=bool(cfg.model.pretrained),
        frozen=bool(cfg.model.frozen),
    ).to(device)
    backbone.eval()

    # Post-init assertion ------------------------------------------------------
    with torch.no_grad():
        dummy = backbone(torch.zeros(1, 3, 224, 224, device=device))
    assert (
        dummy.shape[1] == cfg.model.feature_dim
    ), f"Backbone output dim {dummy.shape[1]} != expected {cfg.model.feature_dim}"

    if cfg.method == "STVM-LapShot":
        classifier: torch.nn.Module = model_lib.STVMLapShot().to(device)
    elif cfg.method == "STDA-LapShot":
        classifier = model_lib.STDALapShot().to(device)
    else:
        raise ValueError(f"Unsupported method: {cfg.method}")
    classifier.eval()

    # ------------------------- Episode iterator ------------------------------
    episode_iter = preprocess.get_episode_loader(cfg.dataset, seed=2026)

    # ------------------------- Evaluation loop -------------------------------
    total_correct = 0
    total_queries = 0
    episode_times: List[float] = []

    n_ways = int(cfg.dataset.n_ways)
    n_shots_first = (
        cfg.dataset.n_shots[0]
        if isinstance(cfg.dataset.n_shots, (list, tuple))
        else cfg.dataset.n_shots
    )
    confusion = torch.zeros(n_ways, n_ways, dtype=torch.long)

    for epi_idx, (sup_imgs, sup_lbls, qry_imgs, qry_lbls) in tqdm(
        enumerate(episode_iter),
        total=int(cfg.dataset.n_episodes),
        desc="Episodes",
        dynamic_ncols=True,
    ):
        # Move to device ------------------------------------------------------
        sup_imgs = sup_imgs.to(device, non_blocking=True)
        qry_imgs = qry_imgs.to(device, non_blocking=True)
        sup_lbls = sup_lbls.to(device, non_blocking=True)
        qry_lbls = qry_lbls.to(device, non_blocking=True)

        # Batch-start assertion (first episode only) --------------------------
        if epi_idx == 0:
            expected_sup = n_ways * int(n_shots_first)
            expected_qry = n_ways * int(cfg.dataset.n_queries)
            assert sup_imgs.shape[0] == expected_sup, (
                f"Support batch has {sup_imgs.shape[0]} instances, expected {expected_sup}"
            )
            assert qry_imgs.shape[0] == expected_qry, (
                f"Query batch has {qry_imgs.shape[0]} instances, expected {expected_qry}"
            )

        # Forward -------------------------------------------------------------
        with torch.no_grad():
            feat_s = backbone(sup_imgs)
            feat_q = backbone(qry_imgs)
            preds = classifier(feat_s=feat_s, y_s=sup_lbls, feat_q=feat_q)

        # Pre-optimizer gradient integrity (skipped for inference-only models)
        if any(p.requires_grad for p in classifier.parameters()):
            for p in classifier.parameters():
                if p.requires_grad:
                    assert p.grad is not None, "Gradient vanished before optimizer.step()"
                    assert torch.any(p.grad != 0), "Gradient is zero before optimizer.step()"

        # Metrics -------------------------------------------------------------
        episode_times.append(classifier.last_inference_time)
        correct = (preds.cpu() == qry_lbls.cpu()).sum().item()
        acc = correct / qry_lbls.numel()

        total_correct += correct
        total_queries += qry_lbls.numel()

        # Confusion matrix update
        for t, p_ in zip(qry_lbls.cpu().tolist(), preds.cpu().tolist()):
            confusion[t, p_] += 1

        if wandb_run:
            wandb.log({"episode_idx": epi_idx, "episode_acc": acc})

    # ------------------------- Aggregation -----------------------------------
    accuracy = total_correct / max(total_queries, 1)
    ci95 = _wald_ci95(total_correct, total_queries)
    avg_inf_ms = float(np.mean(episode_times) * 1e3) if episode_times else 0.0

    summary = {
        "accuracy": accuracy,
        "ci95": ci95,
        "avg_inference_time_ms": avg_inf_ms,
        "confusion_matrix": confusion.tolist(),
    }

    print(
        f"[RESULT] {cfg.run.run_id} — acc={accuracy*100:.2f}% ± {ci95*100:.2f}% | "
        f"avg_inf={avg_inf_ms:.2f} ms",
        flush=True,
    )

    if wandb_run:
        for k, v in summary.items():
            wandb_run.summary[k] = v
        wandb_run.finish()


if __name__ == "__main__":  # pragma: no cover
    main()
