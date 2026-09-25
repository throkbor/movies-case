# 30 made-up films for next year: score them with the three models and let the model spend $300M on them
import numpy as np
import pandas as pd

from backtest import BUDGET, PRED_COLS, profit_and_risk
from model import BASICS, CAST, TRACK, tune_and_fit

PENALTY = 0.25  # the risk penalty i went with (see RISK_PENALTIES in backtest.py)
LAST_YEAR = 2016

# name, budget, month, sequel, genres, flags, track record
PROFILES = [
    ("Franchise action sequel", 200e6, 6, 1, ["action", "adventure"], [], "strong"),
    ("Superhero origin film", 150e6, 5, 0, ["action", "adventure", "science_fiction"], ["is_superhero", "is_adaptation"], "strong"),
    ("Animated family sequel", 100e6, 7, 1, ["animation", "family", "comedy"], [], "strong"),
    ("Original sci-fi", 120e6, 11, 0, ["science_fiction", "action"], [], "typical"),
    ("Mid-budget comedy", 50e6, 6, 0, ["comedy"], [], "typical"),
    ("Adapted drama (novel)", 40e6, 11, 0, ["drama"], ["is_adaptation"], "typical"),
    ("Romantic comedy", 30e6, 2, 0, ["romance", "comedy"], [], "typical"),
    ("Original horror", 15e6, 10, 0, ["horror"], [], "typical"),
    ("Horror sequel", 10e6, 10, 1, ["horror"], [], "typical"),
    ("Indie drama", 5e6, 9, 0, ["drama"], ["is_independent"], "newcomer"),
    ("Original action thriller", 80e6, 3, 0, ["action", "thriller"], [], "typical"),
    ("Family adaptation (book)", 90e6, 12, 0, ["family", "adventure", "fantasy"], ["is_adaptation"], "strong"),
    ("Biopic drama", 25e6, 10, 0, ["drama", "history"], ["is_adaptation"], "typical"),
    ("Low-budget thriller", 8e6, 8, 0, ["thriller"], [], "newcomer"),
    ("Fantasy adventure adaptation", 180e6, 12, 0, ["fantasy", "adventure"], ["is_adaptation"], "strong"),
    ("Original animated film", 90e6, 11, 0, ["animation", "family", "comedy"], [], "typical"),
    ("Superhero sequel", 180e6, 5, 1, ["action", "adventure", "science_fiction"], ["is_superhero", "is_adaptation"], "strong"),
    ("Sci-fi sequel", 160e6, 7, 1, ["science_fiction", "action", "adventure"], [], "strong"),
    ("Live-action family remake", 140e6, 3, 0, ["family", "fantasy", "adventure"], ["is_adaptation"], "strong"),
    ("Action comedy sequel", 90e6, 7, 1, ["action", "comedy"], [], "typical"),
    ("War drama", 70e6, 11, 0, ["war", "drama", "history"], [], "typical"),
    ("Musical", 60e6, 12, 0, ["music", "drama"], [], "typical"),
    ("Western", 50e6, 12, 0, ["western", "drama"], [], "typical"),
    ("Mystery thriller (novel)", 45e6, 10, 0, ["mystery", "thriller"], ["is_adaptation"], "typical"),
    ("Crime thriller", 35e6, 9, 0, ["crime", "thriller"], [], "typical"),
    ("YA romance (novel)", 20e6, 2, 0, ["romance", "drama"], ["is_adaptation"], "typical"),
    ("Teen comedy", 15e6, 8, 0, ["comedy"], [], "typical"),
    ("Low-budget sci-fi", 10e6, 4, 0, ["science_fiction", "thriller"], [], "newcomer"),
    ("Micro-budget horror", 4e6, 1, 0, ["horror"], [], "newcomer"),
    ("Documentary", 4e6, 6, 0, ["documentary"], [], "newcomer"),
]
LEVELS = {"strong": 0.75, "typical": 0.5, "newcomer": 0.25}


def track_record(recent, who, level):
    # a made-up film's director / cast / studio / franchise record = that percentile of recent films that had one
    with_history = recent[recent[f"{who}_n_prior"] > 0]
    return {c: 0 if c.endswith("no_history") else with_history[c].quantile(LEVELS[level])
            for c in TRACK if c.startswith(who + "_")}


def no_history(who):
    return {c: (1 if c.endswith("no_history") else 0 if c.endswith("n_prior") else np.nan)
            for c in TRACK if c.startswith(who + "_")}


def make_profiles(d, genres):
    recent = d[d["year"].between(2010, LAST_YEAR)]
    rows = []
    for name, budget, month, sequel, film_genres, flags, level in PROFILES:
        row = {"name": name, "budget_real": budget, "log_budget": np.log10(budget), "runtime": 107,
               "year": LAST_YEAR + 2, "month": month, "is_sequel": sequel, "is_english": 1, "is_us": 1,
               "n_countries": 1, "n_companies": 3, "is_adaptation": 0, "is_superhero": 0,
               "is_independent": 0, "is_3d": 0}
        row.update({f: 1 for f in flags})
        row.update({g: int(g[len("genre_"):] in film_genres) for g in genres})

        # newcomers: director and cast have no history, and the studio is small (25th percentile)
        for who in ["dir", CAST]:
            row.update(no_history(who) if level == "newcomer" else track_record(recent, who, level))
        row.update(track_record(recent, "studio", level))
        row.update(track_record(recent, "franchise", level) if sequel else no_history("franchise"))
        rows.append(row)
    return pd.DataFrame(rows)


def best_slate(profiles, penalty):
    # checks every full combo that fits in $300M, and skips a branch once it can't beat the best combo so far
    films = profiles.assign(ratio=profiles["exp_profit"] / profiles["budget_real"]).sort_values("ratio", ascending=False)
    budget, profit, var = films["budget_real"].values, films["exp_profit"].values, films["sd_profit"].values ** 2
    best = {"score": -np.inf, "picked": []}

    def most_extra_profit(i, room):
        # the most profit the remaining films could add if you could buy fractions of them, so it's never too low
        extra = 0
        for j in range(i, len(films)):
            if profit[j] <= 0 or room <= 0:
                break
            take = min(1, room / budget[j])
            extra += take * profit[j]
            room -= take * budget[j]
        return extra

    def search(i, spent, total_profit, total_var, picked):
        score = total_profit - penalty * np.sqrt(total_var)
        # same rule as the backtest: the $300M gets spent, so a slate only counts once nothing else fits
        rest = np.delete(budget, picked)
        full = rest.size == 0 or rest.min() > BUDGET - spent
        if full and score > best["score"]:
            best["score"], best["picked"] = score, picked
        if i == len(films):
            return
        # adding films only adds risk, so this is the best any combo from here could possibly score
        if total_profit + most_extra_profit(i, BUDGET - spent) - penalty * np.sqrt(total_var) <= best["score"]:
            return
        if spent + budget[i] <= BUDGET:
            search(i + 1, spent + budget[i], total_profit + profit[i], total_var + var[i], picked + [i])
        search(i + 1, spent, total_profit, total_var, picked)

    search(0, 0, 0, 0, [])
    return profiles.loc[films.index[best["picked"]]]


def main():
    d = pd.read_csv("data/processed/movies_clean.csv")
    d = d[d["year"] <= LAST_YEAR]
    genres = [c for c in d.columns if c.startswith("genre_")]
    features = BASICS + TRACK + genres
    profiles = make_profiles(d, genres)

    # same three models as the walk-forward, but trained on everything up to 2016
    for bar, col in PRED_COLS.items():
        model = tune_and_fit(d[features], (d["roi"] > bar).astype(int))
        profiles[col] = model.predict_proba(profiles[features])[:, 1]

    profiles = profit_and_risk(profiles, d)
    slate = best_slate(profiles, PENALTY)
    profiles["funded"] = profiles.index.isin(slate.index)

    cols = ["name", "budget_real", "pred", "pred_2.5x", "pred_5x", "exp_multiple", "exp_profit", "sd_profit", "funded"]
    print(profiles[cols].sort_values("exp_multiple", ascending=False).round(2).to_string())
    print(f"\nfunded: {profiles['funded'].sum()} films, ${slate['budget_real'].sum() / 1e6:.0f}M, "
          f"expected profit ${slate['exp_profit'].sum() / 1e6:.0f}M, sd ${np.sqrt((slate['sd_profit'] ** 2).sum()) / 1e6:.0f}M")
    profiles[cols].to_csv("data/processed/future_profiles.csv", index=False)


if __name__ == "__main__":
    main()
