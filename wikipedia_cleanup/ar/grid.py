import itertools
import math
from pathlib import Path

from tqdm.auto import tqdm

from wikipedia_cleanup.ar.template_predictor import AssociationRulesTemplatePredictor
from wikipedia_cleanup.predict import TrainAndPredictFramework

n_files = math.ceil(585 / 3)
n_jobs = 4
input_path = Path("data/new_custom/")

MIN_SUPPORT = [0.01, 0.02, 0.03, 0.04, 0.05]
MIN_CONFIDENCE = [0.6, 0.7, 0.8, 0.9]
MIN_TEMPLATE_SUPPORT = [0.0, 0.001]
VAL_SIZE = [0.3, 0.2, 0.1]
VAL_PRECISION = [0.6, 0.7, 0.8, 0.9]
TRANSACTION_FREQ = ["D", "W"]

for vars in tqdm(
    itertools.product(
        MIN_SUPPORT,
        MIN_CONFIDENCE,
        MIN_TEMPLATE_SUPPORT,
        VAL_SIZE,
        VAL_PRECISION,
        TRANSACTION_FREQ,
    ),
    total=len(MIN_SUPPORT)
    * len(MIN_CONFIDENCE)
    * len(MIN_TEMPLATE_SUPPORT)
    * len(VAL_SIZE)
    * len(VAL_PRECISION)
    * len(TRANSACTION_FREQ),
):
    (
        min_support,
        min_confidence,
        min_template_support,
        val_size,
        val_precision,
        transaction_freq,
    ) = vars
    model = AssociationRulesTemplatePredictor(
        min_support=min_support,
        min_confidence=min_confidence,
        min_template_support=min_template_support,
        val_size=val_size,
        val_precision=val_precision,
        transaction_freq=transaction_freq,
    )
    framework = TrainAndPredictFramework(
        model, group_key=["infobox_key", "property_name"]
    )
    framework.load_data(input_path, n_files, n_jobs)
    framework.fit_model()
    print(framework.test_model(predict_subset=1, save_results=False))
