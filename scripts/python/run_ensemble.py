"""Run ensemble.
"""
import numpy as np

from janestreet.setup_env import setup_environment
from janestreet.pipeline import FullPipeline, PipelineCV, PipelineEnsemble
from janestreet.models.nn import NN
from janestreet.data_processor import DataProcessor
from janestreet.tracker import WandbTracker

TRACK = False

MODEL_NAME = "ensemble"
COMMENT = ""
CATEGORY = "model_ver27_ts"
N_SPLITS = 2
START = 1000
MODEL_NAMES = [
    "gru_2.0_700_cv",
    "gru_2.1_700_cv",
    "gru_2.2_700_cv",
    "gru_3.0_700_cv",
    "gru_3.1_700_cv",
    "gru_3.2_700_cv",
]
WEIGHTS = np.array([1.0]*len(MODEL_NAMES))/ len(MODEL_NAMES)
REFIT_MODELS = [True] * len(MODEL_NAMES)

setup_environment(TRACK)

data_processor = DataProcessor(MODEL_NAME, skip_days=START)
df = data_processor.get_train_data()

print(MODEL_NAMES)
print(WEIGHTS)
print(REFIT_MODELS)

models = []
for i, model_name in enumerate(MODEL_NAMES):
    pipeline = FullPipeline(
        NN(),
        name=model_name,
        run_name="full",
        load_model=True,
        features=None,
        refit=True,
        change_lr=False,
    )
    models.append(pipeline)
pipeline = PipelineEnsemble(models, WEIGHTS, REFIT_MODELS)

if TRACK:
    params = {}
    params["n_splits"] = N_SPLITS
    params["model_type"] = "ensemble"
    params["models"] = MODEL_NAMES
    params["weights"] = WEIGHTS
    params["n_models"] = len(MODEL_NAMES)
    wandb_tracker = WandbTracker(
        MODEL_NAME,
        params,
        category=CATEGORY,
        comment=COMMENT
    )
    wandb_tracker.init_run([])
else:
    wandb_tracker = None

cv = PipelineCV(pipeline, wandb_tracker, n_splits=N_SPLITS)
scores = cv.fit(df, verbose=True)

if wandb_tracker is not None:
    wandb_tracker.finish()
