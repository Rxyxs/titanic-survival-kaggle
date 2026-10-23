import math

import pandas as pd
import pytest

from titanic.features import (base_features, group_features, passenger_meta, title_of,
                              woman_boy_rule)


def frame():
    # Ticket A: mother + son + daughter. Ticket B: single man. Ticket C: woman alone.
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5],
        "Name": ["Smith, Mrs. Ann", "Smith, Master. Tom", "Smith, Miss. Sue",
                 "Jones, Mr. Bob", "Brown, Mlle. Eve"],
        "Sex": ["female", "male", "female", "male", "female"],
        "Pclass": [3, 3, 3, 1, 2],
        "Age": [35.0, None, 8.0, 50.0, None],
        "SibSp": [0, 1, 1, 0, 0], "Parch": [2, 1, 1, 0, 0],
        "Ticket": ["A", "A", "A", "B", "C"],
        "Fare": [30.0, 30.0, 30.0, 80.0, None],
        "Cabin": [None, None, None, "C85", None],
        "Embarked": ["S", "S", "S", None, "C"],
    })


def test_titles():
    assert title_of("Smith, Mrs. Ann") == "Mrs"
    assert title_of("Brown, Mlle. Eve") == "Miss"
    assert title_of("Rothes, the Countess. of (Lucy)") == "Mrs"
    assert title_of("Byles, Rev. Thomas") == "Rare"


def test_base_features_have_no_missing_values():
    X = base_features(frame())
    assert not X.isna().any().any()
    assert X.loc[0, "ticket_count"] == 3
    assert X.loc[0, "log_fare_pp"] == pytest.approx(math.log1p(10.0))
    assert X.loc[1, "age_missing"] == 1


def test_group_feature_excludes_own_label():
    meta = passenger_meta(frame())
    labels = pd.Series([0, 1, 0], index=[0, 1, 2])  # mother died, son lived, daughter died
    G = group_features(meta, labels)
    # Son's mates: mother 0, daughter 0 -> rate 0, own label (1) not used.
    assert G.loc[1, "grp_wb_surv"] == 0
    assert G.loc[1, "grp_wb_known"] == 2
    # Including self leaks the passenger's own outcome.
    G_leak = group_features(meta, labels, exclude_self=False)
    assert G_leak.loc[1, "grp_wb_surv"] == pytest.approx(1 / 3)


def test_group_feature_only_uses_given_labels():
    meta = passenger_meta(frame())
    G = group_features(meta, pd.Series([1], index=[0]))  # only the mother is known
    assert G.loc[2, "grp_wb_known"] == 1 and G.loc[2, "grp_wb_surv"] == 1
    assert G.loc[0, "grp_wb_known"] == 0 and G.loc[0, "grp_wb_surv"] == 0.5
    assert G.loc[3, "grp_known"] == 0  # man on his own ticket


def test_woman_boy_rule():
    meta = passenger_meta(frame())
    # Mother and daughter died -> the boy is predicted to die (gender model), and a woman
    # whose woman/boy mates all died is predicted to die.
    G = group_features(meta, pd.Series([0, 0], index=[0, 2]))
    pred = woman_boy_rule(meta, G)
    assert pred.tolist() == [0, 0, 0, 0, 1]
    # Mother survived -> the boy is predicted to survive.
    G = group_features(meta, pd.Series([1], index=[0]))
    assert woman_boy_rule(meta, G).loc[1] == 1
