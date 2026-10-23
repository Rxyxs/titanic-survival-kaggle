"""Which passengers count as a group for the woman-boy rule?

Compares three groupings on the same folds (paired), picks the one with the best
cross-validated accuracy, and writes only that one as a submission. The choice is made
before submitting, so the leaderboard plays no part in it.

    python -m titanic.rule_groups
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import ROOT, load
from .features import GROUPINGS, group_features, passenger_meta, woman_boy_rule
from .models import outer_folds

REPORTS = ROOT / "reports"
SUBS = ROOT / "submissions"
N_REPEATS = 20


def describe(meta: pd.DataFrame, y: pd.Series) -> dict:
    sizes = meta["group"].value_counts()
    test = meta.loc[~meta.index.isin(y.index)]
    wb = meta[meta["woman_or_boy"] == 1]
    wb_train_groups = set(wb.loc[wb.index.isin(y.index), "group"])
    test_wb = test[test["woman_or_boy"] == 1]
    return {
        "groups_with_2plus": int((sizes > 1).sum()),
        "passengers_in_groups": int(sizes[sizes > 1].sum()),
        "largest_group": int(sizes.max()),
        "test_women_boys_with_labelled_wb_mate": int(test_wb["group"].isin(wb_train_groups).sum()),
    }


def main() -> None:
    df, y, train_idx, test_idx = load()
    metas = {g: passenger_meta(df, g) for g in GROUPINGS}
    folds = list(outer_folds(y, N_REPEATS))

    acc = {g: [] for g in GROUPINGS}
    bets = {g: [] for g in GROUPINGS}
    for tr, va in folds:
        yv = y.loc[va].to_numpy()
        for g, meta in metas.items():
            pred = woman_boy_rule(meta, group_features(meta, y.loc[tr])).loc[va].to_numpy()
            acc[g].append(float((pred == yv).mean()))
            b = pred != meta.loc[va, "female"].to_numpy()
            bets[g].append((int(b.sum()), int((pred[b] == yv[b]).sum())))

    out = {}
    for g in GROUPINGS:
        a = np.asarray(acc[g])
        d = a - np.asarray(acc["ticket"])
        rep = d.reshape(N_REPEATS, 5).mean(1)
        n_bets = sum(b for b, _ in bets[g])
        out[g] = {
            "cv_mean": round(float(a.mean()), 4),
            "diff_vs_ticket": round(float(d.mean()), 4),
            "diff_repeats_better_or_equal": int((rep >= 0).sum()),
            "diff_repeat_std": round(float(rep.std(ddof=1)), 4),
            "bets_per_418": round(n_bets / (N_REPEATS * len(y)) * 418, 1),
            "bet_precision": round(sum(r for _, r in bets[g]) / n_bets, 3),
            **describe(metas[g], y),
        }
    best = max(GROUPINGS, key=lambda g: out[g]["cv_mean"])
    out["chosen"] = best
    out["n_repeats"] = N_REPEATS

    meta = metas[best]
    pred = woman_boy_rule(meta, group_features(meta, y)).loc[test_idx]
    ticket_pred = woman_boy_rule(metas["ticket"], group_features(metas["ticket"], y)).loc[test_idx]
    out["test_predictions_changed_vs_ticket"] = int((pred != ticket_pred).sum())
    if best != "ticket":
        pd.DataFrame({"PassengerId": df.loc[test_idx, "PassengerId"].to_numpy(),
                      "Survived": pred.to_numpy().astype(int)}) \
          .to_csv(SUBS / f"woman_boy_rule_{best}.csv", index=False)
    (REPORTS / "rule_groups.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
