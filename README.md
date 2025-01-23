# Kaggle Jane Street Real-Time Market Data Forecasting

Solution for the [Jane Street 2024 Kaggle competition](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview).

A detailed description can be found in [`solution.md`](solution.md) and in [Kaggle discussion](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556542).

## Requirements

- ~100GB RAM
- 12GB GPU RAM

## Usage

1. Install requirements from [`pyproject.toml`](pyproject.toml).
2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data).
3. Set paths and other config variables in [`janestreet/config.py`](janestreet/config.py).

### Scripts

- [`run_cv.py`](scripts/python/run_cv.py) - Estimate model on cross-validation.
- [`run_full.py`](scripts/python/run_full.py) - Estimate model for the final submission (on the whole sample).
- [`run_ensemble.py`](scripts/python/run_ensemble.py) - Evaluate ensemble of models on CV.
- [`run_test_gap.py`](scripts/python/run_test_gap.py) - Test model on a sample of the last 200 dates with a gap of 200 dates.

#### Additional scripts

- [`monitor_kaggle.py`](scripts/python/monitor_kaggle.py) - Monitor kaggle submissions and send notifications when completed.
- [`update_kaggle.py`](scripts/python/update_kaggle.py) - Push code and models to Kaggle datasets to be used in submission.
