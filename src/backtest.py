# if a studio had $300M every 2 years and let the model pick, how would it have done?
# uses the walk-forward predictions, so every film was scored by a model that never saw it
import numpy as np
import pandas as pd

BUDGET = 300e6  # 2017 dollars
N_RANDOM = 1000
BIG = 100e6  # "blockbuster" = budget of $100M or more
CAP = 50  # a few films made 100x+, so for averages and spreads anything over 50x counts as 50x
RISK_PENALTIES = [0, 0.25, 0.75, 1, 1.5]  # how many dollars of expected profit i'd give up to cut $1 of slate std dev
PRED_COLS = {1: "pred", 2.5: "pred_2.5x", 5: "pred_5x"}


def band_averages(train, bars=(1, 2.5, 5)):
    # average (and average squared) return in each band, from the training years.
    # split into budget thirds so a blockbuster that clears 5x is compared with other blockbusters, not tiny breakout hits
    edges = train["budget_real"].quantile([1 / 3, 2 / 3]).values
    group = np.searchsorted(edges, train["budget_real"])
    band = pd.cut(train["roi"], [0, *bars, np.inf], labels=False).values
    capped = train["roi"].clip(upper=CAP)
    mean = capped.groupby([group, band]).mean().unstack().values
    sq = (capped ** 2).groupby([group, band]).mean().unstack().values
    return edges, mean, sq


def profit_and_risk(films, train, bars=(1, 2.5, 5)):
    # each film gets a chance of landing in each band: under 1x, 1-2.5x, 2.5-5x, 5x+
    edges, mean, sq = band_averages(train, bars)
    group = np.searchsorted(edges, films["budget_real"])
    band_mean, band_sq = mean[group], sq[group]

    clear = np.column_stack([np.ones(len(films))] + [films[PRED_COLS[b]] for b in bars] + [np.zeros(len(films))])
    chances = (clear[:, :-1] - clear[:, 1:]).clip(0)
    chances = chances / chances.sum(axis=1, keepdims=True)

    mean_multiple = (chances * band_mean).sum(axis=1)
    sd_multiple = np.sqrt((chances * band_sq).sum(axis=1) - mean_multiple ** 2)
    return films.assign(exp_profit=films["budget_real"] * (mean_multiple - 1),
                        sd_profit=films["budget_real"] * sd_multiple,
                        exp_multiple=mean_multiple)


def risk_adjusted_slate(films, penalty):
    # keep adding whichever film raises (expected profit - penalty * slate std dev) the most per dollar.
    # it's a shortcut, not the exact best slate: too many films to check every combo.
    # films are treated as independent, so the slate's variance is just the sum of theirs
    spent, profit, variance, picked = 0, 0, 0, []
    left = films
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
        train = d[d["year"] < block]
        one = profit_and_risk(films, train, bars=(1,))
        films = profit_and_risk(films, train)
        scored.append(films.assign(exp_multiple_1x=one["exp_multiple"]))

        # do the 2.5x and 5x models pick better films than the 1x model alone? same penalty, same rule
        rows.append({"block": block, "strategy": "1x_model_only", "draw": 0, **result(risk_adjusted_slate(one, 0.25))})
        for penalty in RISK_PENALTIES:
            slate = risk_adjusted_slate(films, penalty)
            rows.append({"block": block, "strategy": f"model_risk_{penalty}", "draw": 0, **result(slate),
                         "predicted_sd": np.sqrt((slate["sd_profit"] ** 2).sum())})
            picks.append(slate.assign(block=block, penalty=penalty))

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

    summary = per_block.groupby("strategy").agg(
        avg_multiple=("multiple", "mean"), sd_multiple=("multiple", "std"), worst_multiple=("multiple", "min"),
        avg_films=("n_films", "mean"))
    print("\nacross blocks:\n" + summary.round(2).to_string())

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
