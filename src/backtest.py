# if a studio had $300M every 2 years and let the model pick, how would it have done?
# uses the walk-forward predictions, so every film was scored by a model that never saw it
import numpy as np
import pandas as pd

BUDGET = 300e6  # 2017 dollars
N_RANDOM = 1000
BIG = 100e6  # "blockbuster" = budget of $100M or more
CAP = 50  # a few films made 100x+. for averages and spreads, anything over 50x counts as 50x
RISK_PENALTIES = [0, 0.25, 0.5, 1]  # how many dollars of expected profit i'd give up to cut $1 of slate std dev
PRED_COLS = {1: "pred", 2.5: "pred_2.5x", 5: "pred_5x"}  # which model's prediction goes with which bar


def profit_and_risk(films, train_roi, bars=(1, 2.5, 5)):
    # the models give each film a chance of landing in each band. with all three: under 1x, 1-2.5x, 2.5-5x, 5x+.
    # with just the 1x model it's only two bands: under 1x and over 1x.
    # the average and spread of each band come from the training years
    bands = pd.cut(train_roi, [0, *bars, np.inf])
    capped = train_roi.clip(upper=CAP)
    band_mean, band_sq = capped.groupby(bands).mean().values, (capped ** 2).groupby(bands).mean().values

    # chance of clearing each bar -> chance of landing between each pair of bars
    clear = np.column_stack([np.ones(len(films))] + [films[PRED_COLS[b]] for b in bars] + [np.zeros(len(films))])
    chances = (clear[:, :-1] - clear[:, 1:]).clip(0)
    chances = chances / chances.sum(axis=1, keepdims=True)

    mean_multiple = chances @ band_mean
    sd_multiple = np.sqrt(chances @ band_sq - mean_multiple ** 2)
    films = films.copy()
    films["exp_profit"] = films["budget_real"] * (mean_multiple - 1)
    films["sd_profit"] = films["budget_real"] * sd_multiple
    films["exp_multiple"] = mean_multiple
    return films


def risk_adjusted_slate(films, penalty):
    # keep adding whichever film raises (expected profit - penalty * slate std dev) the most per dollar.
    # this is a shortcut, not the exact best slate (too many films to check every combo), so the backtest is approximate
    # films are treated as independent, so the slate's variance is just the sum of the films' variances
    spent, profit, variance, picked = 0, 0, 0, []
    left = films.copy()
    while True:
        left = left[left["budget_real"] <= BUDGET - spent]
        if left.empty:
            break
        new_profit = profit + left["exp_profit"]
        new_variance = variance + left["sd_profit"] ** 2
        gain = (new_profit - penalty * np.sqrt(new_variance)) - (profit - penalty * np.sqrt(variance))
        # no stopping early: the brief is to spend the $300M, so it always takes the best (or least bad) next film
        best = (gain / left["budget_real"]).idxmax()
        spent += left.loc[best, "budget_real"]
        profit, variance = new_profit[best], new_variance[best]
        picked.append(best)
        left = left.drop(best)
    return films.loc[picked]


def fill_slate(films):
    # go down the list and take every film that still fits in the budget
    spent, picked = 0, []
    for i, budget in films["budget_real"].items():
        if spent + budget <= BUDGET:
            spent += budget
            picked.append(i)
    return films.loc[picked]


def result(slate):
    return {
        "n_films": len(slate),
        "spend": slate["budget_real"].sum(),
        "revenue": slate["revenue_real"].sum(),
        "multiple": slate["revenue_real"].sum() / slate["budget_real"].sum(),
        "hit_rate": slate["profitable"].mean(),
    }


def main():
    d = pd.read_csv("data/processed/movies_clean.csv")
    p = pd.read_csv("data/processed/predictions_cast.csv")
    p["block"] = (p["year"] // 2 * 2).astype(int)
    rng = np.random.default_rng(0)
    rows, picks, scored = [], [], []

    for block, films in p.groupby("block"):
        train_roi = d.loc[d["year"] < block, "roi"]
        one = profit_and_risk(films, train_roi, bars=(1,))
        films = profit_and_risk(films, train_roi)
        scored.append(films.assign(exp_multiple_1x=one["exp_multiple"]))

        # does adding the 2.5x and 5x models pick better films than the 1x model alone? same penalty, same rule
        rows.append({"block": block, "strategy": "1x_model_only", "draw": 0, **result(risk_adjusted_slate(one, 0.25))})
        for penalty in RISK_PENALTIES:
            slate = risk_adjusted_slate(films, penalty)
            rows.append({"block": block, "strategy": f"model_risk_{penalty}", "draw": 0, **result(slate),
                         "predicted_sd": np.sqrt((slate["sd_profit"] ** 2).sum())})
            picks.append(slate.assign(block=block, penalty=penalty))

        # random picks and random blockbuster picks, many times, to get a range
        big = films[films["budget_real"] >= BIG]
        for draw in range(N_RANDOM):
            rows.append({"block": block, "strategy": "random", "draw": draw,
                         **result(fill_slate(films.sample(frac=1, random_state=rng)))})
            rows.append({"block": block, "strategy": "blockbuster", "draw": draw,
                         **result(fill_slate(big.sample(frac=1, random_state=rng)))})

    r = pd.DataFrame(rows)
    r["profit"] = r["revenue"] - r["spend"]
    per_block = r.groupby(["strategy", "block"])[["multiple", "profit", "n_films"]].median()
    print("revenue / budget of the $300M slate, median over draws:\n"
          + per_block["multiple"].unstack(0).round(2).to_string())

    # how each strategy did across the 9 blocks: average, how much it swung, and the worst block
    summary = per_block.groupby("strategy").agg(
        avg_multiple=("multiple", "mean"), sd_multiple=("multiple", "std"), worst_multiple=("multiple", "min"),
        avg_films=("n_films", "mean"))
    print("\nacross blocks:\n" + summary.round(2).to_string())

    # how well each version's expected multiple lines up with what films actually made (rank correlation)
    scored = pd.concat(scored)
    for col, name in [("exp_multiple_1x", "1x model only"), ("exp_multiple", "all three models")]:
        print(f"rank correlation with actual return, {name}: {scored[col].corr(scored['roi'], method='spearman'):.3f}")
    summary.to_csv("data/processed/backtest_summary.csv")
    scored[["id", "block", "roi", "exp_multiple_1x", "exp_multiple"]].to_csv("data/processed/backtest_scored.csv", index=False)

    picked = pd.concat(picks)
    print("\nmodel picks by risk penalty:")
    print(picked.groupby("penalty").agg(films=("id", "size"), median_budget=("budget_real", "median"),
                                        hit_rate=("profitable", "mean")).round(2).to_string())

    r.to_csv("data/processed/backtest_draws.csv", index=False)
    picked.to_csv("data/processed/backtest_model_picks.csv", index=False)


if __name__ == "__main__":
    main()
