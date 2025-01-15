"""Run model on CV.
"""

from janestreet.setup_env import setup_environment
from janestreet.pipeline import FullPipeline, PipelineCV
from janestreet.models.nn import NN
from janestreet.data_processor import DataProcessor
from janestreet.tracker import WandbTracker
from janestreet.transformers import PolarsTransformer

TRACK = False
COMMENT = ""
CATEGORY = "model_ver27_ts"

MODEL_TYPE = "gru"
NUM = 3
LOAD_MODEL = False
REFIT = True
N_SPLITS = 2
START = 700
TRAIN_SIZE = None

setup_environment(TRACK)

data_processor = DataProcessor(f"{MODEL_TYPE}_{NUM}.{0}_{START}_cv", skip_days=START)
df = data_processor.get_train_data()

features = data_processor.features

params_nn = {
    "model_type": "gru",

    # ### Model 1
    # "hidden_sizes": [250, 150, 150],
    # "dropout_rates": [0.0, 0.0, 0.0],
    # "hidden_sizes_linear": [],
    # "dropout_rates_linear": [],
    # ###

    ### Model 2
    "hidden_sizes": [500],
    "dropout_rates": [0.3, 0.0, 0.0],
    "hidden_sizes_linear": [500, 300],
    "dropout_rates_linear": [0.2, 0.1],
    ###

    "batch_size": 1,
    "early_stopping_patience": 1,
    "lr_refit": 0.0003,
    "lr": 0.0005,

    "epochs": 1000,
    "early_stopping": True,
    "lr_patience": 1000,
    "lr_factor": 0.5,
}

print(features)
print(f"Number of features: {len(features)}")
print(params_nn)

for SEED in range(3):
    MODEL_NAME = f"{MODEL_TYPE}_{NUM}.{SEED}_{700}_cv"

    model = NN(**params_nn, random_seed=SEED)

    pipeline = FullPipeline(
        model,
        preprocessor=PolarsTransformer(features),
        run_name="full",
        name=MODEL_NAME,
        load_model=LOAD_MODEL,
        features=features,
        refit=REFIT,
        change_lr=True,
    )

    wandb_tracker = None
    if TRACK:
        params = dict(params_nn)
        params["n_splits"] = N_SPLITS
        params["seed"] = SEED
        params["start"] = START
        wandb_tracker = WandbTracker(
            MODEL_NAME,
            params,
            category=CATEGORY,
            comment=COMMENT
        )
        wandb_tracker.init_run(features)

    cv = PipelineCV(pipeline, wandb_tracker, n_splits=N_SPLITS, train_size=TRAIN_SIZE)
    scores = cv.fit(df, verbose=True)
    
    if wandb_tracker is not None:
        wandb_tracker.finish()
