# turns movies_metadata.csv + credits.csv into one clean table for the model
import ast

import numpy as np
import pandas as pd

BASE_YEAR = 2017  # all dollars get converted to 2017 dollars
MIN_DOLLARS = 10_000  # budgets or revenues under this are basically always unit errors
MIN_YEAR = 1970
MIN_BUDGET = 1e6  # 2017 dollars. films under $1M are too small to matter for a $300M studio


def parse(text):
    # the list columns are stored as python-looking strings
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []


def names(text):
    return [item["name"] for item in parse(text)]


def load_movies():
    m = pd.read_csv("data/raw/movies_metadata.csv", low_memory=False)
    m["id"] = pd.to_numeric(m["id"], errors="coerce")
    m["budget"] = pd.to_numeric(m["budget"], errors="coerce")
    m = m.dropna(subset=["id"]).drop_duplicates("id")
    m["release_date"] = pd.to_datetime(m["release_date"], errors="coerce")
    m["year"] = m["release_date"].dt.year
    m["month"] = m["release_date"].dt.month

    keep = (
        (m["status"] == "Released")
        & (m["budget"] >= MIN_DOLLARS)
        & (m["revenue"] >= MIN_DOLLARS)
        & (m["year"] >= MIN_YEAR)
        & m["runtime"].between(40, 300)  # a runtime of 0 is really just missing
    )
    # foreign films where revenue is under 1% of budget. the revenue is probably just the us box office
    us_only_revenue = (m["original_language"] != "en") & (m["revenue"] < 0.01 * m["budget"])
    keep = keep & ~us_only_revenue
    return m[keep].copy()


def add_money(d):
    cpi = pd.read_csv("data/cpi_raw.csv", parse_dates=["observation_date"])
    cpi = cpi.groupby(cpi["observation_date"].dt.year)["CPIAUCNS"].mean()
    to_2017_dollars = d["year"].map(cpi[BASE_YEAR] / cpi)

    d["budget_real"] = d["budget"] * to_2017_dollars
    d["revenue_real"] = d["revenue"] * to_2017_dollars
    d["log_budget"] = np.log10(d["budget_real"])
    d["roi"] = d["revenue"] / d["budget"]
    d["profitable"] = (d["revenue"] > d["budget"]).astype(int)
    d["hit_2_5x"] = (d["revenue"] > 2.5 * d["budget"]).astype(int)
    return d


def add_metadata(d):
    d["genres"] = d["genres"].apply(names)
    d["companies"] = d["production_companies"].apply(names)
    d["countries"] = d["production_countries"].apply(names)

    # tmdb adds collections after the fact, so this can't be used as "is a franchise film" on its own.
    # the first star wars is in a collection too. is_sequel gets made from it later, looking only backward
    d["collection"] = d["belongs_to_collection"].apply(lambda x: [parse(x)["name"]] if isinstance(x, str) else [])
    d["is_english"] = (d["original_language"] == "en").astype(int)
    d["is_us"] = d["countries"].apply(lambda c: "United States of America" in c).astype(int)
    d["n_countries"] = d["countries"].apply(len)
    d["n_companies"] = d["companies"].apply(len)

    all_genres = sorted({g for genres in d["genres"] for g in genres})
    for g in all_genres:
        d["genre_" + g.lower().replace(" ", "_")] = d["genres"].apply(lambda genres: g in genres).astype(int)
    return d


def add_credits(d):
    cr = pd.read_csv("data/raw/credits.csv")
    cr["id"] = pd.to_numeric(cr["id"], errors="coerce")
    cr = cr.dropna(subset=["id"]).drop_duplicates("id")
    cr = cr[cr["id"].isin(d["id"])]

    cast = cr["cast"].apply(parse)
    crew = cr["crew"].apply(parse)
    billed = cast.apply(lambda c: [p["name"] for p in sorted(c, key=lambda p: p["order"])])
    cr["top_cast"] = billed.str[:3]
    cr["top5_cast"] = billed.str[:5]  # only for a comparison run against top 3
    cr["directors"] = crew.apply(lambda c: [p["name"] for p in c if p["job"] == "Director"])
    # no cast or crew size: tmdb fills in credits more for popular films, so those counts sneak popularity in

    d = d.merge(cr[["id", "top_cast", "top5_cast", "directors"]], on="id", how="left")
    for col in ["top_cast", "top5_cast", "directors"]:
        d[col] = d[col].apply(lambda x: x if isinstance(x, list) else [])
    return d


def add_keywords(d):
    kw = pd.read_csv("data/raw/keywords.csv")
    kw["id"] = pd.to_numeric(kw["id"], errors="coerce")
    kw = kw.dropna(subset=["id"]).drop_duplicates("id")
    kw["words"] = kw["keywords"].apply(names)
    d = d.merge(kw[["id", "words"]], on="id", how="left")
    words = d["words"].apply(lambda x: x if isinstance(x, list) else [])

    # any "based on ..." tag (novel, comic, video game, tv series, toy, ...) plus biography and remake
    d["is_adaptation"] = words.apply(
        lambda ws: any(w.startswith("based on") or w in ("biography", "remake") for w in ws)).astype(int)
    d["is_superhero"] = words.apply(lambda ws: "superhero" in ws).astype(int)
    d["is_independent"] = words.apply(lambda ws: "independent film" in ws).astype(int)
    d["is_3d"] = words.apply(lambda ws: "3d" in ws).astype(int)
    return d.drop(columns="words")


def add_track_record(d, col, prefix):
    # for each film, look at how its directors / cast / studios / franchise did on films that came out before it.
    # it only ever looks backward, so nothing from the future leaks in
    d = d.sort_values("release_date").reset_index(drop=True)
    past = {}  # name -> {film id: (profitable, revenue, roi)} for their films so far
    stats, rows = [], []

    # films that come out the same day go together: work out all their stats first, then add them to past,
    # so two same-day films can't see each other's results
    for _, day in d.groupby("release_date"):
        for i, film in day.iterrows():
            # merging the dicts means a film two of them were in only counts once
            prior = {}
            for person in film[col]:
                prior.update(past.get(person, {}))

            if prior:
                hits, revenues, rois = zip(*prior.values())
                stats.append([len(prior), np.mean(hits), np.median(revenues), np.mean(revenues),
                              np.median(rois), np.mean(rois)])
            else:
                stats.append([0, np.nan, np.nan, np.nan, np.nan, np.nan])
            rows.append(i)

        for _, film in day.iterrows():
            for person in film[col]:
                past.setdefault(person, {})[film["id"]] = (film["profitable"], film["revenue_real"], film["roi"])

    stat_names = ["n_prior", "hit_rate", "median_revenue", "mean_revenue", "median_roi", "mean_roi"]
    stats = pd.DataFrame(stats, index=rows, columns=[prefix + "_" + n for n in stat_names])
    stats[prefix + "_no_history"] = (stats[prefix + "_n_prior"] == 0).astype(int)
    return pd.concat([d, stats], axis=1)


def main():
    d = load_movies()
    d = add_money(d)
    d = d[d["budget_real"] >= MIN_BUDGET].copy()  # before track records, so they only count films we keep
    d = add_metadata(d)
    d = add_credits(d)
    d = add_keywords(d)
    d = add_track_record(d, "directors", "dir")
    d = add_track_record(d, "top_cast", "cast")
    d = add_track_record(d, "top5_cast", "cast5")
    d = add_track_record(d, "companies", "studio")
    d = add_track_record(d, "collection", "franchise")
    d["is_sequel"] = (d["franchise_n_prior"] > 0).astype(int)

    for col in ["genres", "companies", "countries", "directors", "top_cast", "top5_cast", "collection"]:
        d[col] = d[col].str.join("|")
    raw_cols = ["production_companies", "production_countries", "spoken_languages", "belongs_to_collection",
                "overview", "tagline", "poster_path", "homepage", "video", "adult"]
    d.drop(columns=raw_cols).to_csv("data/processed/movies_clean.csv", index=False)

    print(f"{len(d)} films, {d.year.min():.0f}-{d.year.max():.0f}")
    print(f"profitable: {d.profitable.mean():.3f}, over 2.5x: {d.hit_2_5x.mean():.3f}")


if __name__ == "__main__":
    main()
