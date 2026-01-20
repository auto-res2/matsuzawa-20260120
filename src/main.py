"""Main orchestrator that launches `src.train` as a subprocess.
Reads Hydra config first (to get run_id, mode, results_dir) and forwards the
request with appropriate overrides.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hydra  # noqa: E402, pylint: disable=C0413
from omegaconf import DictConfig  # noqa: E402, pylint: disable=C0413

# -----------------------------------------------------------------------------
# Hydra main ------------------------------------------------------------------
# -----------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="../config", config_name="config")  # type: ignore[misc]
def main(cfg: DictConfig):  # noqa: D401
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'")

    overrides = [
        f"run={cfg.run.run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]

    if cfg.mode == "trial":
        overrides += [
            "wandb.mode=disabled",
            "dataset.n_episodes=2",
            "optuna.n_trials=0",
            "training.epochs=1",
        ]

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        "--config-path",
        "../config",
        "--config-name",
        "config",
    ] + overrides

    env = dict(os.environ, HYDRA_FULL_ERROR="1")
    print("[Main] Launching subprocess:\n  ", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":  # pragma: no cover
    main()
