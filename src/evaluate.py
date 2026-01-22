import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import wandb
from omegaconf import OmegaConf
from scipy import stats

# -----------------------------------------------------------------------------
# Load global WandB settings
# -----------------------------------------------------------------------------

def _load_global_wandb_cfg() -> Dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    cfg = OmegaConf.load(root / "config" / "config.yaml")
    return {"entity": cfg.wandb.entity, "project": cfg.wandb.project}

# -----------------------------------------------------------------------------
# Simple CLI parser (KEY=VALUE)
# -----------------------------------------------------------------------------

def _parse_cli() -> Dict[str, str]:
    if len(sys.argv) < 3:
        raise ValueError("Usage: python -m src.evaluate results_dir=PATH run_ids='[ ]'")
    kv = {}
    for arg in sys.argv[1:]:
        if "=" not in arg:
            raise ValueError(f"Malformed arg {arg}")
        k, v = arg.split("=", 1)
        kv[k] = v
    if {"results_dir", "run_ids"} - kv.keys():
        raise ValueError("Missing required arguments")
    return kv

# -----------------------------------------------------------------------------
# Per-run processing
# -----------------------------------------------------------------------------

def _export_run(run: wandb.apis.public.Run, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = run.history()
    summary = dict(run.summary._json_dict)
    config = dict(run.config)

    (out_dir / "metrics.json").write_text(
        json.dumps({"history": hist.to_dict(orient="list"), "summary": summary, "config": config}, indent=2)
    )

    # Learning curves (if present)
    if {"train_acc_epoch", "val_acc"}.issubset(hist.columns):
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=hist.index, y=hist["train_acc_epoch"], label="train")
        sns.lineplot(x=hist.index, y=hist["val_acc"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Learning Curves – {run.id}")
        plt.tight_layout()
        fig_p = out_dir / f"{run.id}_learning_curve.pdf"
        plt.savefig(fig_p)
        plt.close()
        print(fig_p)

    # Confusion matrix
    if "confusion_matrix" in summary:
        cm = np.array(summary["confusion_matrix"])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, cmap="Blues", annot=False)
        plt.title(f"Confusion Matrix – {run.id}")
        plt.tight_layout()
        fig_p = out_dir / f"{run.id}_confusion_matrix.pdf"
        plt.savefig(fig_p)
        plt.close()
        print(fig_p)

# -----------------------------------------------------------------------------
# Aggregate metrics across runs
# -----------------------------------------------------------------------------

def _aggregate(runs: List[wandb.apis.public.Run], results_dir: Path):
    primary_metric = "accuracy"
    agg: Dict = {"primary_metric": primary_metric, "metrics": {}}

    for r in runs:
        for k, v in r.summary.items():
            if isinstance(v, (float, int)):
                agg["metrics"].setdefault(k, {})[r.id] = float(v)

    proposed = [r for r in runs if "proposed" in r.id.lower() or "adaptive" in r.id.lower()]
    baseline = [r for r in runs if "comparative" in r.id.lower() or "baseline" in r.id.lower()]

    def _best(rs):
        if not rs:
            return None, 0.0
        best = max(rs, key=lambda x: float(x.summary.get(primary_metric, 0.0)))
        return best.id, float(best.summary.get(primary_metric, 0.0))

    best_prop_id, best_prop_val = _best(proposed)
    best_base_id, best_base_val = _best(baseline)

    agg["best_proposed"] = {"run_id": best_prop_id, "value": best_prop_val}
    agg["best_baseline"] = {"run_id": best_base_id, "value": best_base_val}

    # For metrics that should be minimized (loss, error), we invert sign
    maximize = primary_metric.lower() not in {"loss", "error", "ece", "perplexity"}
    if maximize:
        gap_pct = (
            (best_prop_val - best_base_val) / best_base_val * 100.0 if best_base_val != 0 else 0.0
        )
    else:
        gap_pct = (
            (best_base_val - best_prop_val) / best_base_val * 100.0 if best_base_val != 0 else 0.0
        )
    agg["gap"] = gap_pct

    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    (comp_dir / "aggregated_metrics.json").write_text(json.dumps(agg, indent=2))

    # Bar chart for primary metric
    if primary_metric in agg["metrics"]:
        m = agg["metrics"][primary_metric]
        plt.figure(figsize=(8, 4))
        sns.barplot(x=list(m.keys()), y=list(m.values()), palette="viridis")
        plt.ylabel(primary_metric)
        plt.title("Primary Metric Comparison")
        for i, (k, v) in enumerate(m.items()):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig_p = comp_dir / "comparison_accuracy_bar_chart.pdf"
        plt.savefig(fig_p)
        plt.close()
        print(fig_p)

        labels = ["proposed" if k == best_prop_id else "baseline" for k in m.keys()]
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=labels, y=list(m.values()))
        sns.swarmplot(x=labels, y=list(m.values()), color="black", size=3)
        plt.ylabel(primary_metric)
        plt.title("Distribution – Primary Metric")
        plt.tight_layout()
        fig_p = comp_dir / "comparison_accuracy_box_plot.pdf"
        plt.savefig(fig_p)
        plt.close()
        print(fig_p)

        # Significance test if multiple runs each side
        if len(proposed) >= 2 and len(baseline) >= 2:
            prop_scores = [float(r.summary.get(primary_metric, 0.0)) for r in proposed]
            base_scores = [float(r.summary.get(primary_metric, 0.0)) for r in baseline]
            t_stat, p_val = stats.ttest_ind(prop_scores, base_scores, equal_var=False)
            (comp_dir / "significance_test.json").write_text(
                json.dumps({"t_statistic": float(t_stat), "p_value": float(p_val)}, indent=2)
            )
            print(comp_dir / "significance_test.json")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cli = _parse_cli()
    res_dir = Path(cli["results_dir"]).expanduser().resolve()
    run_ids = json.loads(cli["run_ids"])

    wb_cfg = _load_global_wandb_cfg()
    api = wandb.Api()

    runs: List[wandb.apis.public.Run] = []
    for rid in run_ids:
        run = api.run(f"{wb_cfg['entity']}/{wb_cfg['project']}/{rid}")
        runs.append(run)
        _export_run(run, res_dir / rid)

    _aggregate(runs, res_dir)
