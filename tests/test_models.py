import numpy as np
import pandas as pd

from titanic.features import base_features, passenger_meta
from titanic.models import group_labels, make_X, outer_folds


def synthetic(n=60, seed=0):
    rng = np.random.default_rng(seed)
    tickets = [f"T{i // 3}" for i in range(n)]  # groups of three
    sex = rng.choice(["male", "female"], n)
    df = pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Name": [f"X{i}, {'Mrs' if s == 'female' else 'Mr'}. A" for i, s in enumerate(sex)],
        "Sex": sex, "Pclass": rng.integers(1, 4, n), "Age": rng.uniform(1, 70, n),
        "SibSp": 0, "Parch": 0, "Ticket": tickets, "Fare": rng.uniform(5, 100, n),
        "Cabin": None, "Embarked": "S",
    })
    y = pd.Series(rng.integers(0, 2, n), index=df.index)
    return df, y


def test_honest_group_features_ignore_validation_labels():
    df, y = synthetic()
    meta, base = passenger_meta(df), base_features(df)
    tr, va = next(outer_folds(y, n_repeats=1))
    X = make_X(base, meta, group_labels(y, tr, "honest"), "honest")
    y_flipped = y.copy()
    y_flipped.loc[va] = 1 - y_flipped.loc[va]
    X2 = make_X(base, meta, group_labels(y_flipped, tr, "honest"), "honest")
    pd.testing.assert_frame_equal(X, X2)


def test_leaky_mode_sees_validation_labels():
    df, y = synthetic()
    meta, base = passenger_meta(df), base_features(df)
    tr, va = next(outer_folds(y, n_repeats=1))
    y_flipped = y.copy()
    y_flipped.loc[va] = 1 - y_flipped.loc[va]
    X = make_X(base, meta, group_labels(y, tr, "minus_self"), "minus_self")
    X2 = make_X(base, meta, group_labels(y_flipped, tr, "minus_self"), "minus_self")
    assert not X.equals(X2)


def test_outer_folds_partition_each_repeat():
    _, y = synthetic()
    folds = list(outer_folds(y, n_repeats=2))
    assert len(folds) == 10
    for r in range(2):
        va_all = np.concatenate([va for _, va in folds[r * 5:(r + 1) * 5]])
        assert sorted(va_all) == list(y.index)
