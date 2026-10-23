"""Model zoo with small hyperparameter grids and a nested cross-validation harness."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .features import group_features

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42


def zoo() -> dict[str, tuple[object, dict]]:
    """name -> (estimator, grid). Scaled pipelines for distance/linear models."""
    return {
        "logreg": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            {"logisticregression__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        ),
        "svm": (
            make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
            {"svc__C": [0.3, 1.0, 3.0], "svc__gamma": ["scale", 0.01, 0.03]},
        ),
        "knn": (
            make_pipeline(StandardScaler(), KNeighborsClassifier()),
            {"kneighborsclassifier__n_neighbors": [5, 9, 15, 25]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
            {"max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 3, 5]},
        ),
        "lightgbm": (
            LGBMClassifier(n_estimators=300, learning_rate=0.03, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8,
                           random_state=SEED, verbose=-1, n_jobs=1),
            {"num_leaves": [4, 8, 16], "min_child_samples": [10, 20, 40]},
        ),
        "xgboost": (
            XGBClassifier(n_estimators=300, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, random_state=SEED, n_jobs=1,
                          eval_metric="logloss"),
            {"max_depth": [2, 3, 4, 6], "min_child_weight": [1, 5]},
        ),
        "catboost": (
            CatBoostClassifier(iterations=400, learning_rate=0.05, random_seed=SEED,
                               verbose=0, thread_count=1, allow_writing_files=False),
            {"depth": [3, 4, 6]},
        ),
    }


def make_X(base: pd.DataFrame, meta: pd.DataFrame, labels: pd.Series | None,
           group_mode: str) -> pd.DataFrame:
    """Feature matrix for all rows. ``group_mode``: none | honest | minus_self | with_self."""
    if group_mode == "none":
        return base
    exclude_self = group_mode != "with_self"
    return pd.concat([base, group_features(meta, labels, exclude_self=exclude_self)], axis=1)


@dataclass
class FoldResult:
    acc: list[float] = field(default_factory=list)
    inner_best: list[float] = field(default_factory=list)
    params: list[dict] = field(default_factory=list)


def outer_folds(y: pd.Series, n_repeats: int, seed: int = SEED):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    idx = y.index.to_numpy()
    for tr, va in rskf.split(idx, y.to_numpy()):
        yield idx[tr], idx[va]


def group_labels(y: pd.Series, tr: np.ndarray, group_mode: str) -> pd.Series:
    """Which labels the group features may use while scoring a fold.

    honest     -> training fold only (what the test set will see)
    minus_self -> every train label except the passenger's own (validation mates leak)
    with_self  -> every train label, own label included (gross leak)
    """
    return y.loc[tr] if group_mode == "honest" else y


def nested_cv(base, meta, y, models: dict, group_mode: str, n_repeats: int = 3,
              n_jobs: int = -1) -> tuple[dict[str, FoldResult], dict[str, list[float]]]:
    """Outer repeated 5-fold, inner 5-fold grid search. Also scores two fixed ensembles
    (soft vote and majority vote over every model), whose membership is set in advance."""
    res = {name: FoldResult() for name in models}
    ens = {"ensemble_soft": [], "ensemble_hard": []}
    for tr, va in outer_folds(y, n_repeats):
        X = make_X(base, meta, group_labels(y, tr, group_mode), group_mode)
        probs = []
        for name, (est, grid) in models.items():
            gs = GridSearchCV(est, grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                              scoring="accuracy", n_jobs=n_jobs)
            gs.fit(X.loc[tr], y.loc[tr])
            p = gs.predict_proba(X.loc[va])[:, 1]
            probs.append(p)
            res[name].acc.append(float(((p >= 0.5).astype(int) == y.loc[va].to_numpy()).mean()))
            res[name].inner_best.append(float(gs.best_score_))
            res[name].params.append(gs.best_params_)
        P = np.vstack(probs)
        yv = y.loc[va].to_numpy()
        ens["ensemble_soft"].append(float(((P.mean(0) >= 0.5) == yv).mean()))
        ens["ensemble_hard"].append(float((((P >= 0.5).mean(0) > 0.5) == yv).mean()))
    return res, ens


def fit_full(base, meta, y, test_idx, models: dict, group_mode: str, n_jobs: int = -1):
    """Tune each model on all labelled passengers and return test probabilities."""
    X = make_X(base, meta, y if group_mode != "none" else None, group_mode)
    out, params = {}, {}
    for name, (est, grid) in models.items():
        gs = GridSearchCV(est, grid, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2,
                                                                random_state=SEED),
                          scoring="accuracy", n_jobs=n_jobs)
        gs.fit(X.loc[y.index], y)
        out[name] = gs.predict_proba(X.loc[test_idx])[:, 1]
        params[name] = gs.best_params_
    return out, params
