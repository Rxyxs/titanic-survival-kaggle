"""Feature engineering for the Titanic passenger list.

Two kinds of features:

* ``base_features`` uses only passenger attributes (no labels), so it is computed once
  on train + test together. Using the test rows' attributes (ticket counts, age medians)
  is allowed: they are features, not answers.
* ``group_features`` uses the survival labels of a passenger's ticket mates. It must be
  computed from the labels the model is allowed to see; inside cross-validation that means
  the training fold only. ``labels`` is the set of known outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Mrs", "Countess": "Mrs",
    "Dona": "Mrs", "the Countess": "Mrs",
}


def title_of(name: str) -> str:
    raw = name.split(",", 1)[1].split(".", 1)[0].strip()
    return TITLE_MAP.get(raw, "Rare")


GROUPINGS = ("ticket", "family", "union")


def family_key(df: pd.DataFrame) -> pd.Series:
    """Surname + class + ticket without its last character + fare + port.

    Two passengers with the same surname are only the same family if they also travelled
    on matching tickets in the same class from the same port; common surnames (Smith,
    Johnson) would otherwise merge strangers."""
    surname = df["Name"].str.split(",", n=1).str[0].str.strip()
    return (surname + "|" + df["Pclass"].astype(str) + "|" + df["Ticket"].str[:-1] + "|"
            + df["Fare"].astype(str) + "|" + df["Embarked"].astype(str))


def _components(*keys: pd.Series) -> pd.Series:
    """Connected components of passengers linked by any shared key (union-find)."""
    parent = list(range(len(keys[0])))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key in keys:
        first = {}
        for i, k in enumerate(key.to_numpy()):
            if k in first:
                a, b = find(i), find(first[k])
                if a != b:
                    parent[a] = b
            else:
                first[k] = i
    return pd.Series([f"g{find(i)}" for i in range(len(parent))], index=keys[0].index)


def group_key(df: pd.DataFrame, how: str = "ticket") -> pd.Series:
    """ticket: same ticket. family: same family key. union: linked by either."""
    if how == "ticket":
        return df["Ticket"]
    if how == "family":
        return family_key(df)
    if how == "union":
        return _components(df["Ticket"], family_key(df))
    raise ValueError(how)


def passenger_meta(df: pd.DataFrame, grouping: str = "ticket") -> pd.DataFrame:
    """Columns needed by both feature sets and by the rule-based model."""
    meta = pd.DataFrame(index=df.index)
    meta["title"] = df["Name"].map(title_of)
    meta["female"] = (df["Sex"] == "female").astype(int)
    meta["boy"] = (meta["title"] == "Master").astype(int)
    meta["woman_or_boy"] = ((meta["female"] == 1) | (meta["boy"] == 1)).astype(int)
    meta["group"] = group_key(df, grouping)
    return meta


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-free features for every row of ``df`` (train and test concatenated)."""
    meta = passenger_meta(df)
    X = pd.DataFrame(index=df.index)
    X["pclass"] = df["Pclass"]
    X["female"] = meta["female"]

    age_median = df.groupby([meta["title"], df["Pclass"]])["Age"].transform("median")
    age_median = age_median.fillna(df["Age"].median())
    X["age_missing"] = df["Age"].isna().astype(int)
    X["age"] = df["Age"].fillna(age_median)
    X["child"] = (X["age"] < 14).astype(int)

    X["sibsp"] = df["SibSp"]
    X["parch"] = df["Parch"]
    X["family_size"] = df["SibSp"] + df["Parch"] + 1
    X["alone"] = (X["family_size"] == 1).astype(int)

    ticket_count = df.groupby("Ticket")["Ticket"].transform("count")
    X["ticket_count"] = ticket_count
    fare = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))
    fare = fare.fillna(df["Fare"].median())
    X["log_fare"] = np.log1p(fare)
    X["log_fare_pp"] = np.log1p(fare / ticket_count)

    deck = df["Cabin"].fillna("U").str[0].replace({"T": "U"})
    X["cabin_known"] = (deck != "U").astype(int)
    embarked = df["Embarked"].fillna("S")

    X = pd.concat(
        [
            X,
            pd.get_dummies(meta["title"], prefix="title").astype(int),
            pd.get_dummies(embarked, prefix="emb").astype(int),
            pd.get_dummies(deck, prefix="deck").astype(int),
        ],
        axis=1,
    )
    return X


def group_features(
    meta: pd.DataFrame, labels: pd.Series, exclude_self: bool = True
) -> pd.DataFrame:
    """Survival of each passenger's ticket mates, using only ``labels``.

    ``labels`` is indexed like ``meta`` and holds the outcomes the model may see.
    With ``exclude_self=False`` a passenger's own label leaks into its feature; that
    variant exists only to measure the leak.
    """
    known = meta.loc[labels.index].assign(y=labels.astype(float))

    def mate_stats(mask: pd.Series) -> tuple[pd.Series, pd.Series]:
        sub = known[mask.loc[known.index].astype(bool)]
        s = sub.groupby("group")["y"].agg(["sum", "count"])
        tot_sum = meta["group"].map(s["sum"]).fillna(0.0)
        tot_cnt = meta["group"].map(s["count"]).fillna(0.0)
        if exclude_self:
            own = mask & meta.index.isin(known.index)
            own_y = known["y"].reindex(meta.index).fillna(0.0)
            tot_sum = tot_sum - own_y * own
            tot_cnt = tot_cnt - own.astype(float)
        rate = (tot_sum / tot_cnt.replace(0, np.nan))
        return rate, tot_cnt

    everyone = pd.Series(True, index=meta.index)
    rate_all, n_all = mate_stats(everyone)
    rate_wb, n_wb = mate_stats(meta["woman_or_boy"] == 1)

    G = pd.DataFrame(index=meta.index)
    G["grp_known"] = n_all
    G["grp_surv"] = rate_all.fillna(0.5)
    G["grp_wb_known"] = n_wb
    G["grp_wb_surv"] = rate_wb.fillna(0.5)
    G["grp_wb_all_died"] = ((n_wb > 0) & (rate_wb == 0)).astype(int)
    G["grp_wb_all_lived"] = ((n_wb > 0) & (rate_wb == 1)).astype(int)
    return G


def woman_boy_rule(meta: pd.DataFrame, groups: pd.DataFrame) -> pd.Series:
    """Gender model with one family correction.

    Women survive and men die, except: a woman whose known woman/boy ticket mates all
    died is predicted to die, and a boy (title Master) whose known woman/boy ticket mates
    all survived is predicted to survive.
    """
    pred = meta["female"].copy()
    pred[(meta["female"] == 1) & (groups["grp_wb_all_died"] == 1)] = 0
    pred[(meta["boy"] == 1) & (groups["grp_wb_all_lived"] == 1)] = 1
    return pred.astype(int)
