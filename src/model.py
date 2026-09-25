# walk-forward xgboost: train on every film before a 2 year block, predict that block, then move forward 2 years
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

CAST = "cast"  # switch to "cast5" to compare top 3 vs top 5 billed actors
LAST_TEST_YEAR = 2016  # the data stops mid 2017, and late 2017 revenue probably wasn't final yet

BASICS = [
    "log_budget", "runtime", "year", "month",
    "is_sequel", "is_english", "is_us", "n_countries", "n_companies",
    "is_adaptation", "is_superhero", "is_independent", "is_3d",
]
TRACK_STATS = ["n_prior", "hit_rate", "median_revenue", "mean_revenue", "median_roi", "mean_roi", "no_history"]
TRACK = [f"{who}_{stat}" for who in ["dir", CAST, "studio", "franchise"] for stat in TRACK_STATS]

# the main model is "did it make its budget back". the 2.5x and 5x ones just say how big the win is
BARS = [1, 2.5, 5]

SEARCH_SPACE = {
    "max_depth": randint(2, 7),
    "learning_rate": loguniform(0.01, 0.3),
    "n_estimators": randint(100, 600),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 20),
    "reg_lambda": loguniform(0.1, 10),
}


def score(y, p, baseline):
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc_flop": average_precision_score(1 - y, 1 - p),  # how well it picks out the flops
        "accuracy": accuracy_score(y, p > 0.5),
        "baseline_accuracy": (y == baseline).mean(),
        "brier": brier_score_loss(y, p),
    }


def tune_and_fit(X, y):
    # the tuning cv is time-ordered too, so it never trains on films later than the ones it tests on
    search = RandomizedSearchCV(
        xgb.XGBClassifier(eval_metric="logloss", random_state=0),
        SEARCH_SPACE, n_iter=30, scoring="roc_auc",
        cv=TimeSeriesSplit(n_splits=3), random_state=0, n_jobs=-1,
    )
    return search.fit(X, y)


def shap_values(model, X):
    # the last column xgboost returns is the baseline, not a feature
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(contribs[:, :-1], columns=X.columns, index=X.index)


def main():
    d = pd.read_csv("data/processed/movies_clean.csv").sort_values("release_date").reset_index(drop=True)
    genres = [c for c in d.columns if c.startswith("genre_")]
    X = d[BASICS + TRACK + genres]
    y = d["profitable"]
    print(f"{len(d)} films, {X.shape[1]} features, cast = {CAST}")

    d["pred"] = np.nan
    d["baseline"] = np.nan
    folds, shaps = [], []
    for start in range(2000, LAST_TEST_YEAR + 1, 2):
        end = min(start + 1, LAST_TEST_YEAR)
        years = f"{start}-{end}" if end > start else str(start)
        train = d["year"] < start
        test = d["year"].between(start, end)

        search = tune_and_fit(X[train], y[train])
        d.loc[test, "pred"] = search.predict_proba(X[test])[:, 1]
        d.loc[test, "baseline"] = int(y[train].mean() >= 0.5)
        shaps.append(shap_values(search.best_estimator_, X[test]))

        folds.append({"years": years, "n_train": train.sum(), "n_test": test.sum(),
                      **score(y[test], d.loc[test, "pred"], d.loc[test, "baseline"]), "params": search.best_params_})

        for bar in BARS[1:]:
            y_bar = (d["roi"] > bar).astype(int)
            d.loc[test, f"pred_{bar}x"] = tune_and_fit(X[train], y_bar[train]).predict_proba(X[test])[:, 1]
            folds[-1][f"roc_auc_{bar}x"] = roc_auc_score(y_bar[test], d.loc[test, f"pred_{bar}x"])

        print(f"{years}: auc {folds[-1]['roc_auc']:.3f} (2.5x {folds[-1]['roc_auc_2.5x']:.3f}, "
              f"5x {folds[-1]['roc_auc_5x']:.3f}) on {test.sum()} films")

    tested = d[d["pred"].notna()]
    print("\nall folds together:")
    for name, value in score(tested["profitable"], tested["pred"], tested["baseline"]).items():
        print(f"  {name}: {value:.3f}")
    for bar in BARS[1:]:
        print(f"  roc_auc {bar}x: {roc_auc_score(tested['roi'] > bar, tested[f'pred_{bar}x']):.3f}")

    shap = pd.concat(shaps)
    importance = shap.abs().mean().sort_values(ascending=False)
    print("\ntop features (mean |shap|):\n" + importance.head(15).round(3).to_string())

    shap.insert(0, "id", d.loc[shap.index, "id"])
    tested.to_csv(f"data/processed/predictions_{CAST}.csv", index=False)
    shap.to_csv(f"data/processed/shap_{CAST}.csv", index=False)
    pd.DataFrame(folds).to_csv(f"data/processed/fold_metrics_{CAST}.csv", index=False)
    importance.to_csv(f"data/processed/feature_importance_{CAST}.csv", header=["mean_abs_shap"])


if __name__ == "__main__":
    main()
