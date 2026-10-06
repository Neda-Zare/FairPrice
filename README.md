# FairPrice

**Is this used car priced fairly?** FairPrice learns from real listings on [Divar](https://divar.ir) (Iran's largest classifieds site) and answers two questions about any car:

1. **What should it cost?** Give it a brand/model, production year and mileage, and it suggests a price.
2. **Is the asking price cheap, fair or expensive?** Give it the same details plus a price, and it tells you whether the price is below, within or above the standard range.

Built as a university **Linear Algebra** project. The regression is written from scratch on top of NumPy: no scikit-learn, just the linear algebra.

---

## How it works

```
Divar listings ──► scrape_divar.py ──► data/cars.csv ──► regression.py ──► model.pkl ──► predict.py
 (real ads)         (collect data)      (cleaned rows)    (fit the model)   (saved)      (price / verdict)
```

### The model

The log of the price is modelled as a linear combination of the features:

$$\log(\text{price}) = X w + \varepsilon$$

Each row of $X$ is one listing. The columns are

- a constant column (the intercept),
- **year** and **mileage** (standardized to zero mean, unit variance),
- one 0/1 column per **make** (first word of the name, e.g. "Peugeot") and per **model** (full name, e.g. "Peugeot 206").

Predicting the log makes the model multiplicative: a car loses a roughly constant *percentage* of its value per year, not a constant amount.

### Solving the least-squares problem

There are far more listings than features, so $Xw = y$ has no exact solution. We take the least-squares solution $\min_w \lVert Xw - y \rVert^2$ (the orthogonal projection of $y$ onto the column space of $X$), computed in two ways that agree to about $10^{-14}$:

| Solver | Idea |
|---|---|
| **Normal equations** | Solve $(X^\top X)\,w = X^\top y$ directly. |
| **QR decomposition** | Factor $X = QR$, then solve the triangular system $R\,w = Q^\top y$ by back-substitution (implemented by hand). More numerically stable, since $X^\top X$ is never formed. |

`test_regression.py` checks both against `numpy.linalg.lstsq`.

### Ridge regularization

Most brands have only a handful of listings, so giving each its own unconstrained weight would overfit. The brand/model weights are therefore shrunk towards zero (ridge regression):

$$\min_w \lVert Xw - y \rVert^2 + \lambda \lVert w_{\text{brand}} \rVert^2$$

This is implemented purely with linear algebra: append the rows $\sqrt{\lambda}\,I$ (on the brand columns) to $X$ and zeros to $y$, then solve the ordinary least-squares problem on the augmented system. The same two solvers work unchanged. Rare brands are pulled toward the average car; common brands keep their own price level.

### Cheap / fair / expensive

The model's typical error $\sigma$ (in log-price units) is estimated by 5-fold cross-validation, i.e. on listings the model has not seen. With predicted price $\hat p$, the **fair range** is

$$\left[\hat p\, e^{-0.5\sigma},\ \hat p\, e^{+0.5\sigma}\right]$$

A price below the range is *cheap*, above it *expensive*, inside it *fair*. The factor `0.5` (`k` in `CarPriceModel.judge`) controls how strict the verdict is.

### Data cleaning

Real listings are messy, so before fitting the model:

- rows with a missing brand, zero/negotiable price, or an impossible year or mileage are dropped;
- listings whose price is a clear outlier (e.g. a fake placeholder price) are trimmed using a robust median-absolute-deviation rule, and the model is refit.

---

## Quick start

Requires Python 3.9+.

```bash
pip install numpy pandas requests
```

### 1. Collect data (skip if you already have `data/cars.csv`)

```bash
python scrape_divar.py --city 1 --pages 40 --out data/cars.csv
```

| Option | Default | Meaning |
|---|---|---|
| `--city` | `1` | Divar numeric city id (`1` = Tehran) |
| `--category` | `cars` | Category slug |
| `--pages` | `20` | Number of result pages (~24 ads each) |
| `--delay` | `2.0` | Seconds between requests (be polite) |
| `--debug` | off | Print the status and start of every response |

The output CSV has the columns `brand, year, mileage_km, engine_cc, automatic, body_damage, price, token`. Years are Solar Hijri (Jalali) and prices are in **millions of toman**. The model currently uses `brand`, `year`, `mileage_km` and `price`.

> Divar's web API is unofficial and can change. If the scraper returns nothing, run it with `--debug` and inspect the saved raw responses in `data/raw/`.

### 2. Train and use the model

Interactive mode (just run it, no arguments):

```bash
python predict.py
```

It trains a model on first run (and whenever the saved one is outdated), then asks for brand/model, year, mileage and, optionally, the asking price.

Or use the command line:

```bash
# train (solver: qr or normal) and print evaluation
python predict.py train --solver qr

# suggest a price
python predict.py price --brand "پژو 206" --year 1395 --mileage 160000

# cheap / fair / expensive?
python predict.py judge --brand "پژو 206" --year 1395 --mileage 160000 --price 1100
```

Example (the program prints Persian; translated here):

```
Suggested price:    1,237 million toman
Standard range:     1,016 to 1,506
Your price (1,100): within the standard range (fair)
```

Brand names must match how they appear in the data (Persian, e.g. `"پژو 206"` = Peugeot 206). Unseen brands fall back to the average car.

### 3. Run the tests

```bash
python test_regression.py
```

Trains with both solvers, prints evaluation metrics, asserts that both match `numpy.linalg.lstsq`, and sanity-checks the cheap/fair/expensive verdicts.

---

## Results

Evaluated on a sample of ~425 Tehran listings, over 30 random 80/20 train/test splits:

| Metric | Value |
|---|---|
| Median absolute error | **≈ 18%** |
| R² on log-price | **≈ 0.7** |
| Typical error σ (log scale) | ≈ 0.4 |

For comparison, with a plain unregularized model that lumps rare brands together, the median error was about 28%.

### Limitations

- **Small dataset.** A few hundred ads; more data is the single most effective improvement.
- **Few features.** Only year, mileage and brand/model. Trim level, options, real body/engine condition and color are not modelled.
- **Asking prices, not sale prices.** Sellers set prices with a lot of noise, so some of the error is irreducible.
- **Expensive and rare brands** (BMW, Lexus, ...) are predicted less accurately, and the fair range has the same relative width for every car, so it is only reliable for clearly cheap or clearly expensive listings.

---

## Project structure

```
FairPrice/
├── scrape_divar.py      # collect listings from Divar into data/cars.csv
├── regression.py        # data cleaning, normal-equation + QR solvers, ridge model, cheap/fair/expensive logic
├── predict.py           # interactive and command-line interface
├── test_regression.py   # solver correctness and evaluation
└── data/
    └── cars.csv         # listings (brand, year, mileage_km, ..., price in million toman)
```

## Data and fair use

The scraper uses Divar's public web endpoints at a low request rate and is intended for educational purposes only. Keep the delay between requests.