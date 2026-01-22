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
        json.dumps(
            {
                "history": hist.to_dict(orient="list"),
                "summary": summary,
                "config": config,
            },
            indent=2,
        )
    )

    # Learning curves (if present)
    if {"train_acc_epoch", "val_acc"}.issubset(hist.columns):
        plt.figure(figsize=(8, 5), dpi=300)
        sns.lineplot(
            x=hist.index, y=hist["train_acc_epoch"], label="Train", linewidth=2
        )
        sns.lineplot(x=hist.index, y=hist["val_acc"], label="Validation", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title(f"Learning Curves – {run.id}", fontsize=14, fontweight="bold")
        # Use consistent, focused Y-axis range for better visibility
        plt.ylim(
            0.7, 1.0
        )  # Consistent scale across all runs, focused on relevant range
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend(fontsize=11, frameon=True, shadow=True)
        plt.tight_layout()
        fig_p = out_dir / f"{run.id}_learning_curve.pdf"
        plt.savefig(fig_p, dpi=300, bbox_inches="tight")
        plt.close()
        print(fig_p)

    # Confusion matrix
    if "confusion_matrix" in summary:
        cm = np.array(summary["confusion_matrix"])
        cifar10_classes = [
            "Airplane",
            "Auto",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck",
        ]
        plt.figure(figsize=(10, 8), dpi=300)
        sns.heatmap(
            cm,
            cmap="Blues",
            annot=True,
            fmt="g",
            cbar_kws={"label": "Count"},
            xticklabels=cifar10_classes,
            yticklabels=cifar10_classes,
            linewidths=0.5,
        )
        plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
        plt.ylabel("True Label", fontsize=12, fontweight="bold")
        plt.title(
            f"Confusion Matrix – {run.id}", fontsize=14, fontweight="bold", pad=20
        )
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        fig_p = out_dir / f"{run.id}_confusion_matrix.pdf"
        plt.savefig(fig_p, dpi=300, bbox_inches="tight")
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

    proposed = [
        r for r in runs if "proposed" in r.id.lower() or "adaptive" in r.id.lower()
    ]
    baseline = [
        r for r in runs if "comparative" in r.id.lower() or "baseline" in r.id.lower()
    ]

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
            (best_prop_val - best_base_val) / best_base_val * 100.0
            if best_base_val != 0
            else 0.0
        )
    else:
        gap_pct = (
            (best_base_val - best_prop_val) / best_base_val * 100.0
            if best_base_val != 0
            else 0.0
        )
    agg["gap"] = gap_pct

    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    (comp_dir / "aggregated_metrics.json").write_text(json.dumps(agg, indent=2))

    # Bar chart for primary metric
    if primary_metric in agg["metrics"]:
        m = agg["metrics"][primary_metric]

        # Determine colors based on run type
        colors = []
        labels_short = []
        for k in m.keys():
            if "proposed" in k.lower() or "adaptive" in k.lower():
                colors.append("#2ecc71")  # Green for proposed
                labels_short.append("Proposed")
            else:
                colors.append("#3498db")  # Blue for baseline
                labels_short.append("Baseline")

        plt.figure(figsize=(10, 6), dpi=300)
        bars = plt.bar(
            range(len(m)),
            list(m.values()),
            color=colors,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
        )
        plt.ylabel(primary_metric.capitalize(), fontsize=13, fontweight="bold")
        plt.xlabel("Run ID", fontsize=13, fontweight="bold")
        plt.title("Primary Metric Comparison", fontsize=15, fontweight="bold", pad=20)

        # Add value labels on top of bars
        for i, (k, v) in enumerate(m.items()):
            plt.text(
                i,
                v + 0.01,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.xticks(range(len(m)), labels_short, fontsize=11)
        plt.ylim(0.7, 0.9)  # More focused Y-axis range
        plt.grid(True, axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        fig_p = comp_dir / "comparison_accuracy_bar_chart.pdf"
        plt.savefig(fig_p, dpi=300, bbox_inches="tight")
        plt.close()
        print(fig_p)

        # Only create box plot if we have multiple runs per category
        labels = ["Proposed" if k == best_prop_id else "Baseline" for k in m.keys()]

        # Count runs per category
        from collections import Counter

        label_counts = Counter(labels)

        if all(count > 1 for count in label_counts.values()):
            # We have multiple runs - create box plot
            plt.figure(figsize=(8, 6), dpi=300)
            sns.boxplot(
                x=labels,
                y=list(m.values()),
                palette={"Baseline": "#3498db", "Proposed": "#2ecc71"},
                width=0.5,
                linewidth=2,
            )
            sns.swarmplot(
                x=labels, y=list(m.values()), color="black", size=6, alpha=0.6
            )
            plt.ylabel(primary_metric.capitalize(), fontsize=13, fontweight="bold")
            plt.xlabel("Method", fontsize=13, fontweight="bold")
            plt.title(
                "Distribution – Primary Metric", fontsize=15, fontweight="bold", pad=20
            )
            plt.grid(True, axis="y", alpha=0.3, linestyle="--")
            plt.tight_layout()
            fig_p = comp_dir / "comparison_accuracy_box_plot.pdf"
            plt.savefig(fig_p, dpi=300, bbox_inches="tight")
            plt.close()
            print(fig_p)
        else:
            # Single run per category - create grouped bar chart instead
            plt.figure(figsize=(8, 6), dpi=300)
            categories = list(set(labels))
            values_by_cat = {cat: [] for cat in categories}
            for label, val in zip(labels, m.values()):
                values_by_cat[label].append(val)

            x_pos = np.arange(len(categories))
            heights = [
                values_by_cat[cat][0] if values_by_cat[cat] else 0 for cat in categories
            ]
            colors_map = {"Baseline": "#3498db", "Proposed": "#2ecc71"}
            bar_colors = [colors_map.get(cat, "#95a5a6") for cat in categories]

            bars = plt.bar(
                x_pos,
                heights,
                color=bar_colors,
                edgecolor="black",
                linewidth=1.2,
                alpha=0.85,
            )
            plt.ylabel(primary_metric.capitalize(), fontsize=13, fontweight="bold")
            plt.xlabel("Method", fontsize=13, fontweight="bold")
            plt.title(
                "Primary Metric Comparison by Method",
                fontsize=15,
                fontweight="bold",
                pad=20,
            )
            plt.xticks(x_pos, categories, fontsize=12)

            # Add value labels
            for i, (cat, h) in enumerate(zip(categories, heights)):
                plt.text(
                    i,
                    h + 0.005,
                    f"{h:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

            plt.ylim(0.7, 0.9)
            plt.grid(True, axis="y", alpha=0.3, linestyle="--")
            plt.tight_layout()
            fig_p = comp_dir / "comparison_accuracy_box_plot.pdf"
            plt.savefig(fig_p, dpi=300, bbox_inches="tight")
            plt.close()
            print(fig_p)

        # Significance test if multiple runs each side
        if len(proposed) >= 2 and len(baseline) >= 2:
            prop_scores = [float(r.summary.get(primary_metric, 0.0)) for r in proposed]
            base_scores = [float(r.summary.get(primary_metric, 0.0)) for r in baseline]
            t_stat, p_val = stats.ttest_ind(prop_scores, base_scores, equal_var=False)
            (comp_dir / "significance_test.json").write_text(
                json.dumps(
                    {"t_statistic": float(t_stat), "p_value": float(p_val)}, indent=2
                )
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
