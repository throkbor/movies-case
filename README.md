# movies-case

Movie profitability analysis on Kaggle's [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

## Data

Download the dataset from Kaggle and unzip the CSVs into `data/raw/`. `data/cpi_raw.csv` is the FRED CPIAUCNS series, used to put all money in 2017 dollars.

## Run

From the repo root:

```
pip install -r requirements.txt
python src/prepare.py    # clean table -> data/processed/movies_clean.csv
python src/model.py      # walk-forward xgboost: revenue over 1x, 2.5x and 5x budget
python src/backtest.py   # $300M slates picked by the model vs random and blockbusters, 2000-2016
python src/future.py     # 30 made-up films for next year, and which ones the model funds
```

Then run `notebooks/eda.ipynb` and `notebooks/model_charts.ipynb`. All charts save to `figures/`.
