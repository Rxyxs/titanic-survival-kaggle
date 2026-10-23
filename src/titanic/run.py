"""End-to-end pipeline: EDA tables, baselines, nested CV, leak study, final fits, submissions.

    python -m titanic.run            # everything (about 20 minutes on a laptop CPU)
    python -m titanic.run --quick    # 1 outer repeat, for a smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, load
from .features import base_features, group_features, passenger_meta, woman_boy_rule
from .models import fit_full, group_labels, make_X, nested_cv, outer_folds, zoo

REPORTS = ROOT / "reports"
SUBS = ROOT / "submissions"


def eda(df: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> dict:
    tr = df.loc[y.index].assign(Survived=y, title=meta.loc[y.index, "title"])
    tr["family"] = pd.cut(tr["SibSp"] + tr["Parch"] + 1, [0, 1, 4, 20],
                          labels=["alone", "2-4", "5+"])
    tr["age_band"] = pd.cut(tr["Age"], [0, 14, 30, 50, 100],
                            labels=["<14", "14-29", "30-49", "50+"])
    def rate(col):
        g = tr.groupby(col, observed=True)["Survived"]
        return {str(k): {"rate": round(float(v), 3), "n": int(n)}
                for k, v, n in zip(g.mean().index, g.mean().values, g.size().values)}
    out = {
        "n_train": int(len(tr)), "n_test": int(len(df) - len(tr)),
        "survival_rate": round(float(y.mean()), 3),
        "missing": {c: int(df[c].isna().sum()) for c in ["Age", "Cabin", "Embarked", "Fare"]},
        "by_sex": rate("Sex"), "by_class": rate("Pclass"), "by_title": rate("title"),
        "by_family": rate("family"), "by_age_band": rate("age_band"),
        "by_sex_class": {f"{s}-{c}": v for (s, c), v in
                         tr.groupby(["Sex", "Pclass"])["Survived"].mean().round(3).items()},
    }
    # How many test passengers share a ticket with someone whose fate is known?
    test_tickets = meta.loc[~meta.index.isin(y.index), "ticket"]
    train_tickets = set(meta.loc[y.index, "ticket"])
    out["test_with_labelled_ticket_mate"] = int(test_tickets.isin(train_tickets).sum())
    return out


def rule_cv(meta: pd.DataFrame, y: pd.Series, n_repeats: int) -> dict:
    """Fold accuracies of three label-free/rule baselines, group features per leak mode."""
    res = {"all_die": [], "gender": [], "rule_honest": [], "rule_minus_self": [],
           "rule_with_self": []}
    for tr, va in outer_folds(y, n_repeats):
        yv = y.loc[va].to_numpy()
        res["all_die"].append(float((yv == 0).mean()))
        res["gender"].append(float((meta.loc[va, "female"].to_numpy() == yv).mean()))
        for mode in ["honest", "minus_self", "with_self"]:
            G = group_features(meta, group_labels(y, tr, mode), exclude_self=mode != "with_self")
            pred = woman_boy_rule(meta, G).loc[va].to_numpy()
            res[f"rule_{mode}"].append(float((pred == yv).mean()))
    return res


def summarize(acc: list[float], n_splits: int = 5) -> dict:
    a = np.asarray(acc)
    reps = a.reshape(-1, n_splits).mean(1)
    return {"mean": round(float(a.mean()), 4), "fold_std": round(float(a.std(ddof=1)), 4),
            "repeat_means": [round(float(r), 4) for r in reps], "folds": [round(x, 4) for x in a]}


def leaky_model(base, meta, y, test_idx):
    """LightGBM trained on group features that include each passenger's own label.

    Its CV looks near-perfect; on the test set the feature cannot contain the answer,
    so the leaderboard shows what the model really learned."""
    from lightgbm import LGBMClassifier
    X = make_X(base, meta, y, "with_self")
    m = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8,
                       min_child_samples=10, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, random_state=42, verbose=-1)
    m.fit(X.loc[y.index], y)
    return m.predict_proba(X.loc[test_idx])[:, 1] >= 0.5


def main(quick: bool = False) -> None:
    REPORTS.mkdir(exist_ok=True)
    SUBS.mkdir(exist_ok=True)
    t0 = time.time()
    df, y, train_idx, test_idx = load()
    meta = passenger_meta(df)
    base = base_features(df)
    n_rep = 1 if quick else 3

    (REPORTS / "eda.json").write_text(json.dumps(eda(df, y, meta), indent=2))
    print("eda done")

    rules = {k: summarize(v) for k, v in rule_cv(meta, y, 1 if quick else 10).items()}
    (REPORTS / "baselines.json").write_text(json.dumps(rules, indent=2))
    print("baselines", {k: v["mean"] for k, v in rules.items()})

    models = zoo()
    cv = {}
    for mode in ["none", "honest"]:
        res, ens = nested_cv(base, meta, y, models, mode, n_repeats=n_rep)
        cv[mode] = {name: {**summarize(r.acc),
                           "inner_best_mean": round(float(np.mean(r.inner_best)), 4),
                           "params": r.params}
                    for name, r in res.items()}
        cv[mode].update({k: summarize(v) for k, v in ens.items()})
        print(mode, {k: v["mean"] for k, v in cv[mode].items()}, f"{time.time()-t0:.0f}s")

    leak_models = {k: models[k] for k in ["logreg", "lightgbm"]}
    for mode in ["minus_self", "with_self"]:
        res, _ = nested_cv(base, meta, y, leak_models, mode, n_repeats=n_rep)
        cv[mode] = {name: summarize(r.acc) for name, r in res.items()}
        print(mode, {k: v["mean"] for k, v in cv[mode].items()})
    (REPORTS / "cv.json").write_text(json.dumps(cv, indent=2))

    # Final fits on all 891 labelled passengers.
    pid = df.loc[test_idx, "PassengerId"].to_numpy()

    def save(name: str, pred) -> None:
        pd.DataFrame({"PassengerId": pid, "Survived": np.asarray(pred).astype(int)}) \
          .to_csv(SUBS / f"{name}.csv", index=False)

    save("gender", meta.loc[test_idx, "female"])
    save("woman_boy_rule", woman_boy_rule(meta, group_features(meta, y)).loc[test_idx])

    final = {}
    for mode in ["none", "honest"]:
        probs, params = fit_full(base, meta, y, test_idx, models, mode)
        tag = "base" if mode == "none" else "group"
        scores = {k: v["mean"] for k, v in cv[mode].items() if k in models}
        best = max(scores, key=scores.get)
        save(f"{best}_{tag}", probs[best] >= 0.5)
        save(f"ensemble_soft_{tag}", np.mean(list(probs.values()), axis=0) >= 0.5)
        final[tag] = {"best_single": best, "params": params,
                      "test_survival_rate": {k: round(float((p >= 0.5).mean()), 3)
                                             for k, p in probs.items()}}
    (REPORTS / "final.json").write_text(json.dumps(final, indent=2, default=str))
    save("lightgbm_leaky", leaky_model(base, meta, y, test_idx))
    print(f"done in {time.time()-t0:.0f}s; submissions in {SUBS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
