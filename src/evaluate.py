"""Independent evaluation & visualisation script.
Retrieves run data from WandB and produces per-run as well as aggregated
analysis artefacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from scipy import stats

PRIMARY_METRIC = "accuracy"  # must exactly match train.py summary key

# -----------------------------------------------------------------------------
# Minimal CLI parsing ----------------------------------------------------------
# -----------------------------------------------------------------------------

def _parse_cli(argv: List[str]) -> Tuple[str, str]:
    toks = argv[1:]
    params: Dict[str, str] = {}
    positional: List[str] = []
    for tok in toks:
        if "=" in tok:
            k, v = tok.split("=", 1)
            params[k.strip()] = v.strip()
        else:
            positional.append(tok)
    if "results_dir" in params and "run_ids" in params:
        return params["results_dir"], params["run_ids"]
    if len(positional) == 2:
        return positional[0], positional[1]
    raise RuntimeError(
        "Unable to parse CLI arguments. Expected 'results_dir' and 'run_ids'."
    )


# -----------------------------------------------------------------------------
# I/O helpers ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2))
    print("[Evaluate] Saved", path)


def _fig_name(topic: str, run_id: str | None = None, *, suffix: str = "pdf") -> str:
    return f"{('comparison' if run_id is None else run_id)}_{topic}.{suffix}"


# -----------------------------------------------------------------------------
# Per-run processing -----------------------------------------------------------
# -----------------------------------------------------------------------------

def _process_run(api: wandb.Api, entity: str, project: str, run_id: str, out_dir: Path) -> Dict[str, Any]:
    print(f"[Evaluate] Fetching {run_id} …")
    run = api.run(f"{entity}/{project}/{run_id}")
    hist_df: pd.DataFrame = run.history(stream="default")
    if hist_df.empty:
        raise RuntimeError(f"Run {run_id} has no history.")

    summary = dict(run.summary)
    config = dict(run.config)

    _ensure_dir(out_dir)
    _save_json(out_dir / "metrics.json", {
        "summary": summary,
        "config": config,
        "history": hist_df.to_dict(orient="list"),
    })

    # ------------------- Learning curve --------------------------------------
    if {"episode_idx", "episode_acc"}.issubset(hist_df.columns):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist_df["episode_idx"], hist_df["episode_acc"], lw=0.8)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Learning curve – {run_id}")
        ax.grid(True, ls="--", lw=0.3)
        fig.tight_layout()
        fp = out_dir / _fig_name("learning_curve", run_id)
        fig.savefig(fp)
        plt.close(fig)
        print("[Evaluate] Generated", fp)

    # ------------------- Confusion matrix ------------------------------------
    if "confusion_matrix" in summary:
        import seaborn as sns  # local import to keep global namespace clean

        conf = np.asarray(summary["confusion_matrix"], dtype=np.int64)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion – {run_id}")
        fig.tight_layout()
        fp = out_dir / _fig_name("confusion_matrix", run_id)
        fig.savefig(fp)
        plt.close(fig)
        print("[Evaluate] Generated", fp)

    return {
        "accuracy": float(summary.get("accuracy", np.nan)),
        "ci95": float(summary.get("ci95", np.nan)),
        "time_ms": float(summary.get("avg_inference_time_ms", np.nan)),
        "episode_acc_series": hist_df["episode_acc"].tolist(),
    }


# -----------------------------------------------------------------------------
# Aggregation & comparison -----------------------------------------------------
# -----------------------------------------------------------------------------

def _aggregate(all_metrics: Dict[str, Dict[str, Any]], out_root: Path) -> None:
    cmp_dir = out_root / "comparison"
    _ensure_dir(cmp_dir)

    by_metric: Dict[str, Dict[str, float]] = {}
    for rid, vals in all_metrics.items():
        for m, v in vals.items():
            if m == "episode_acc_series":
                continue
            by_metric.setdefault(m, {})[rid] = v

    # Identify best proposed / baseline --------------------------------------
    best_prop = {"run_id": None, "value": -np.inf}
    best_base = {"run_id": None, "value": -np.inf}
    for rid, val in by_metric.get(PRIMARY_METRIC, {}).items():
        lower = rid.lower()
        if any(k in lower for k in ("proposed", "stvm", "ours")) and val > best_prop["value"]:
            best_prop = {"run_id": rid, "value": val}
        elif any(k in lower for k in ("comparative", "baseline", "stda")) and val > best_base["value"]:
            best_base = {"run_id": rid, "value": val}

    gap = (
        (best_prop["value"] - best_base["value"]) / best_base["value"] * 100
        if np.isfinite(best_base["value"]) and best_base["value"] else np.nan
    )

    aggregated = {
        "primary_metric": PRIMARY_METRIC,
        "metrics": by_metric,
        "best_proposed": best_prop,
        "best_baseline": best_base,
        "gap": gap,
    }
    _save_json(cmp_dir / "aggregated_metrics.json", aggregated)

    # ------------------- Bar chart ------------------------------------------
    runs = list(by_metric[PRIMARY_METRIC].keys())
    values = [by_metric[PRIMARY_METRIC][r] for r in runs]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=runs, y=values, palette="viridis", ax=ax)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy across runs")
    for i, v in enumerate(values):
        ax.text(i, v + 0.002, f"{v*100:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticklabels(runs, rotation=45, ha="right")
    fig.tight_layout()
    fp = cmp_dir / _fig_name("accuracy_bar_chart")
    fig.savefig(fp)
    plt.close(fig)
    print("[Evaluate] Generated", fp)

    # ------------------- Box plot & significance ----------------------------
    series_df = pd.DataFrame({rid: m["episode_acc_series"] for rid, m in all_metrics.items()})
    long_df = series_df.melt(var_name="run_id", value_name="acc")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="run_id", y="acc", data=long_df, palette="Set2", ax=ax)
    ax.set_ylabel("Episode accuracy")
    ax.set_title("Episode-level accuracy distribution")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fp = cmp_dir / _fig_name("accuracy_boxplot")
    fig.savefig(fp)
    plt.close(fig)
    print("[Evaluate] Generated", fp)

    # Welch t-test between best prop & baseline ------------------------------
    pvalue = float("nan")
    if best_prop["run_id"] and best_base["run_id"]:
        pvalue = stats.ttest_ind(
            series_df[best_prop["run_id"]],
            series_df[best_base["run_id"]],
            equal_var=False,
        ).pvalue
    _save_json(cmp_dir / "statistical_significance.json", {
        "best_proposed": best_prop,
        "best_baseline": best_base,
        "pvalue": pvalue,
    })


# -----------------------------------------------------------------------------
# Entry point ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    results_dir_str, run_ids_str = _parse_cli(sys.argv)
    out_root = Path(results_dir_str).expanduser().resolve()
    _ensure_dir(out_root)

    run_ids: List[str] = json.loads(run_ids_str)

    # Load WandB credentials from global config ------------------------------
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    import yaml  # local import only here for light-weight

    with cfg_path.open("r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)
    entity = base_cfg["wandb"]["entity"]
    project = base_cfg["wandb"]["project"]

    api = wandb.Api()
    all_metrics: Dict[str, Dict[str, Any]] = {}
    for rid in run_ids:
        all_metrics[rid] = _process_run(api, entity, project, rid, out_root / rid)

    _aggregate(all_metrics, out_root)


if __name__ == "__main__":  # pragma: no cover
    main()
