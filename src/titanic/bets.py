"""Every model is the gender rule plus a set of "bets": passengers where it disagrees
with sex alone (a man predicted to survive, a woman predicted to die).

On the test set the leaderboard gives only one number per submission, but because the
gender rule was also submitted, the net value of the bets follows from arithmetic:

    net correct bets = (score(model) - score(gender)) * 418
    precision of the bets = (bets + net) / (2 * bets)

The same quantity in cross-validation comes from out-of-fold predictions. No test label
is used anywhere: only the public scores, which are the competition's own feedback.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .data import ROOT, load
from .features import base_features, group_features, passenger_meta, woman_boy_rule
from .models import make_X, zoo

REPORTS = ROOT / "reports"
SUBS = ROOT / "submissions"
N_TEST = 418


def oof_predictions(base, meta, y, params: dict, n_repeats: int = 3) -> dict[str, np.ndarray]:
    """Out-of-fold 0/1 predictions with the hyperparameters chosen on the full train set,
    group features from training-fold labels only. Returns predictions per repeat stacked."""
    models = zoo()
    preds = {k: [] for k in ["logreg_group", "ensemble_soft_group", "woman_boy_rule"]}
    for r in range(n_repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=100 + r)
        p_lr = pd.Series(0, index=y.index)
        p_ens = pd.Series(0, index=y.index)
        p_rule = pd.Series(0, index=y.index)
        for tr, va in skf.split(y.index, y):
            tr, va = y.index[tr], y.index[va]
            X = make_X(base, meta, y.loc[tr], "honest")
            probs = []
            for name, (est, _) in models.items():
                m = clone(est).set_params(**params[name])
                m.fit(X.loc[tr], y.loc[tr])
                probs.append(m.predict_proba(X.loc[va])[:, 1])
            p_lr.loc[va] = (probs[list(models).index("logreg")] >= 0.5).astype(int)
            p_ens.loc[va] = (np.mean(probs, axis=0) >= 0.5).astype(int)
            G = group_features(meta, y.loc[tr])
            p_rule.loc[va] = woman_boy_rule(meta, G).loc[va].to_numpy()
        preds["logreg_group"].append(p_lr.to_numpy())
        preds["ensemble_soft_group"].append(p_ens.to_numpy())
        preds["woman_boy_rule"].append(p_rule.to_numpy())
    return preds


def bet_stats(pred: np.ndarray, female: np.ndarray, truth: np.ndarray) -> dict:
    bets = pred != female
    n = int(bets.sum())
    right = int((pred[bets] == truth[bets]).sum())
    men_saved = int((bets & (female == 0)).sum())
    return {"bets_per_passenger": round(n / len(pred), 4), "bets": n,
            "precision": round(right / n, 3) if n else None,
            "men_predicted_to_survive": men_saved, "women_predicted_to_die": n - men_saved}


def main() -> None:
    df, y, train_idx, test_idx = load()
    meta = passenger_meta(df)
    base = base_features(df)
    final = json.loads((REPORTS / "final.json").read_text())
    lb = {}
    for r in json.loads((REPORTS / "leaderboard.json").read_text()):
        lb.setdefault(r["name"], float(r["public_score"]))
    female_te = meta.loc[test_idx, "female"].to_numpy()
    female_tr = meta.loc[train_idx, "female"].to_numpy()

    out = {"cv": {}, "test": {}}
    oof = oof_predictions(base, meta, y, final["group"]["params"])
    for name, reps in oof.items():
        stats = [bet_stats(p, female_tr, y.to_numpy()) for p in reps]
        out["cv"][name] = {
            "bets_per_passenger": round(float(np.mean([s["bets_per_passenger"] for s in stats])), 4),
            "precision": round(float(np.mean([s["precision"] for s in stats])), 3),
            "men_predicted_to_survive": round(float(np.mean([s["men_predicted_to_survive"]
                                                             for s in stats])), 1),
            "women_predicted_to_die": round(float(np.mean([s["women_predicted_to_die"]
                                                           for s in stats])), 1),
        }

    for name, score in lb.items():
        if name == "gender":
            continue
        pred = pd.read_csv(SUBS / f"{name}.csv")["Survived"].to_numpy()
        s = bet_stats(pred, female_te, truth=pred)  # truth unknown: counts only
        net = round((score - lb["gender"]) * N_TEST)
        s["net_correct_bets"] = int(net)
        s["precision"] = round((s["bets"] + net) / (2 * s["bets"]), 3) if s["bets"] else None
        s["public_score"] = score
        out["test"][name] = s
    (REPORTS / "bets.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
