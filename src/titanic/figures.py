"""Figures for the README and the project page, built from reports/*.json."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import ROOT

REPORTS = ROOT / "reports"
IMG = ROOT / "docs" / "img"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

NAMES = {
    "gender": "Gender rule", "rule_honest": "Woman-boy rule",
    "logreg": "Logistic regression", "svm": "SVM (RBF)", "knn": "k-NN",
    "random_forest": "Random forest", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "catboost": "CatBoost", "ensemble_soft": "Ensemble (soft vote)",
    "ensemble_hard": "Ensemble (majority)",
}


def load(name):
    return json.loads((REPORTS / name).read_text())


def eda_figure():
    e = load("eda.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
    x = np.arange(3)
    w = 0.38
    for i, (sex, color) in enumerate([("female", BLUE), ("male", ORANGE)]):
        vals = [e["by_sex_class"][f"{sex}-{c}"] for c in (1, 2, 3)]
        bars = a1.bar(x + (i - 0.5) * (w + 0.02), vals, w, color=color, label=sex.capitalize())
        a1.bar_label(bars, labels=[f"{v:.0%}" for v in vals], fontsize=8, color=INK2, padding=2)
    a1.set_xticks(x, ["1st class", "2nd class", "3rd class"])
    a1.set_ylim(0, 1.08)
    a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a1.set_title("Survival by sex and class")
    a1.legend(frameon=False, loc="upper right")

    t = e["by_title"]
    order = sorted(t, key=lambda k: t[k]["rate"])
    vals = [t[k]["rate"] for k in order]
    bars = a2.barh(order, vals, color=BLUE, height=0.6)
    a2.bar_label(bars, labels=[f"{t[k]['rate']:.0%}  (n={t[k]['n']})" for k in order],
                 fontsize=8, color=INK2, padding=3)
    a2.set_xlim(0, 1.15)
    a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.set_title("Survival by title (from the name)")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(IMG / "eda.png", dpi=150)
    plt.close(fig)


def kaggle_scores():
    try:
        rows = load("leaderboard.json")
    except FileNotFoundError:
        return {}
    out = {}
    for r in rows:  # newest first; keep the latest score per submission name
        if r["public_score"] and r["name"] not in out:
            out[r["name"]] = float(r["public_score"])
    return out


def validation_figure(lb_map: dict[str, str]):
    """CV accuracy (folds + mean) per model, with the Kaggle score where submitted.
    ``lb_map`` maps a row key to the submission name that scored it."""
    base, cv = load("baselines.json"), load("cv.json")
    lb = kaggle_scores()
    rows = [("gender", base["gender"]), ("rule_honest", base["rule_honest"])]
    rows += [(k, v) for k, v in cv["honest"].items()]
    rows.sort(key=lambda kv: kv[1]["mean"])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, (k, v) in enumerate(rows):
        reps = v["repeat_means"]
        ax.scatter(reps, [i] * len(reps), s=22, color=BLUE, alpha=0.35, linewidths=0, zorder=2)
        ax.scatter([v["mean"]], [i], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3,
                   label="CV mean (dots: each repeat)" if i == 0 else None)
        sub = lb_map.get(k)
        if sub in lb:
            ax.scatter([lb[sub]], [i], s=80, marker="D", color=ORANGE, edgecolor=SURFACE,
                       linewidth=2, zorder=4, label="Kaggle leaderboard")
    ax.set_yticks(range(len(rows)), [NAMES.get(k, k) for k, _ in rows])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Accuracy")
    ax.set_title("Cross-validation vs. the Kaggle leaderboard (with ticket-group features)")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(IMG / "validation.png", dpi=150)
    plt.close(fig)


def leak_figure():
    base, cv = load("baselines.json"), load("cv.json")
    groups = ["Woman-boy rule", "Logistic regression", "LightGBM"]
    modes = [("honest", "Training-fold labels only", BLUE),
             ("minus_self", "All labels except own", AQUA),
             ("with_self", "All labels incl. own", ORANGE)]
    vals = {
        "honest": [base["rule_honest"]["mean"], cv["honest"]["logreg"]["mean"],
                   cv["honest"]["lightgbm"]["mean"]],
        "minus_self": [base["rule_minus_self"]["mean"], cv["minus_self"]["logreg"]["mean"],
                       cv["minus_self"]["lightgbm"]["mean"]],
        "with_self": [base["rule_with_self"]["mean"], cv["with_self"]["logreg"]["mean"],
                      cv["with_self"]["lightgbm"]["mean"]],
    }
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(groups))
    w = 0.26
    for j, (m, label, color) in enumerate(modes):
        bars = ax.bar(x + (j - 1) * (w + 0.02), vals[m], w, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals[m]], fontsize=8, color=INK2,
                     padding=2)
    ax.set_xticks(x, groups)
    ax.set_ylim(0.7, 1.1)
    ax.set_yticks(np.arange(0.7, 1.01, 0.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("CV accuracy")
    ax.set_title("Which labels may the ticket-group feature use? (5-fold CV)")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(IMG / "leak.png", dpi=150)
    plt.close(fig)


def bets_figure():
    b = load("bets.json")
    keys = ["logreg_group", "ensemble_soft_group", "woman_boy_rule"]
    labels = ["Logistic regression\n+ groups", "Ensemble (soft vote)\n+ groups",
              "Woman-boy rule"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(keys))
    w = 0.36
    for j, (src, name, color) in enumerate([("cv", "Cross-validation (train)", BLUE),
                                             ("test", "Kaggle test set", ORANGE)]):
        vals = [b[src][k]["precision"] for k in keys]
        bars = ax.bar(x + (j - 0.5) * (w + 0.02), vals, w, color=color, label=name)
        if src == "test":
            lab = [f"{v:.0%}  ({b['test'][k]['bets']} bets)" for k, v in zip(keys, vals)]
        else:
            lab = [f"{v:.0%}" for v in vals]
        ax.bar_label(bars, labels=lab, fontsize=8, color=INK2, padding=2)
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(2.62, 0.51, "coin flip", fontsize=8, color=INK2, ha="right", va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share of bets that are right")
    ax.set_title("Bets against the gender rule: how often are they right?")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(IMG / "bets.png", dpi=150)
    plt.close(fig)


def main(lb_map: dict[str, str] | None = None):
    IMG.mkdir(parents=True, exist_ok=True)
    eda_figure()
    bets_figure()
    leak_figure()
    validation_figure(lb_map or {})


if __name__ == "__main__":
    import sys
    # Optional JSON map {row key: submission name} for the leaderboard markers.
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else None)
