"""Run model for final submission.
"""

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from janestreet.setup_env import setup_environment
from janestreet.pipeline import FullPipeline
from janestreet.models.nn import NN
from janestreet.config import COL_DATE, COL_WEIGHT, COL_TARGET
from janestreet.data_processor import DataProcessor
from janestreet.metrics import r2_weighted
from janestreet.transformers import PolarsTransformer


MODEL_TYPE = "gru"
START = 500
NUM = 3
EPOCHS_LS = [8]*10
COMMENT = ""
CATEGORY = "model_ver22_ts"

LOAD_MODEL = False
REFIT = True

setup_environment()

data_processor = DataProcessor(
    f"{MODEL_TYPE}_{NUM}.0_700",
    skip_days=START,
    transformer=PolarsTransformer()
)
df = data_processor.get_train_data()
features = data_processor.features

params_nn = {
    "model_type": "gru",

    ### Model 1
    "hidden_sizes": [250, 150, 150],
    "dropout_rates": [0.0, 0.0, 0.0],
    "hidden_sizes_linear": [],
    "dropout_rates_linear": [],
    ###

    ### Model 2
    # "hidden_sizes": [500],
    # "dropout_rates": [0.3, 0.0, 0.0],
    # "hidden_sizes_linear": [500, 300],
    # "dropout_rates_linear": [0.2, 0.1],
    ###

    "batch_size": 1,
    "early_stopping_patience": 1,
    "lr_refit": 0.0003,
    "lr": 0.0005,

    "epochs": 100,
    "early_stopping": True,
    "lr_patience": 10,
    "lr_factor": 0.5,
}

print(features)
print(f"Number of features: {len(features)}")
print(params_nn)

for SEED in range(3):
    MODEL_NAME = f"{MODEL_TYPE}_{NUM}.{SEED}_700"
    EPOCHS = EPOCHS_LS[SEED]

    print(MODEL_NAME)

    if EPOCHS is not None:
        params_nn["early_stopping"] = False
        params_nn["epochs"] = EPOCHS
        print(f"Running final model with {params_nn['epochs']} epochs.")
    model = NN(**params_nn, random_seed=SEED)

    df_train = df.filter(pl.col(COL_DATE) >= START+200)
    df_valid = df.filter(pl.col(COL_DATE) < START+200)

    pipeline = FullPipeline(
        model,
        preprocessor=None,
        run_name="full",
        name=MODEL_NAME,
        load_model=LOAD_MODEL,
        features=features,
        refit=REFIT,
    )
    pipeline.fit(df_train, df_valid, verbose=True)

    cnt_dates = 0
    preds = []
    dates = np.unique(df_valid.select(pl.col(COL_DATE)).to_series().to_numpy())
    for date_id in tqdm(dates):
        df_valid_date = df.filter(pl.col(COL_DATE) == date_id)
        if pipeline.refit & (cnt_dates > 0):
            df_valid_upd = df.filter(pl.col(COL_DATE) == date_id-1)
            pipeline.update(df_valid_upd)
        preds_i, hidden = pipeline.predict(df_valid_date, n_times=None)
        preds += list(preds_i)
        cnt_dates += 1

    preds = np.array(preds)
    y = df_valid.select(pl.col(COL_TARGET)).to_series().to_numpy()
    weight = df_valid.select(pl.col(COL_WEIGHT)).to_series().to_numpy()
    score = r2_weighted(y, preds, weight)
    print(f"Score: {score:.5f}")
