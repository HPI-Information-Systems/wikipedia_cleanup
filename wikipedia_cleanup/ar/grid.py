import collections
import functools
import itertools
import operator
import resource
from pathlib import Path

from tqdm.auto import tqdm

from wikipedia_cleanup.ar.template_predictor import AssociationRulesTemplatePredictor
from wikipedia_cleanup.predict import TrainAndPredictFramework


def generate_grid(**items):
    grid_generator = itertools.starmap(
        collections.namedtuple("GridEntry", items.keys()),
        itertools.product(*items.values()),
    )
    grid_len = functools.reduce(
        operator.mul, [len(value) for value in items.values()], 1
    )
    return grid_generator, grid_len


def limit_memory(maxsize: int) -> None:
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))


limit_memory(128_000_000_000)
n_files = 585
n_jobs = 6
input_path = Path("data/new_custom/")
# /san2/data/change-exploration/mp2021_wiki_cleanup/new_costum_filtered_format

grid, grid_len = generate_grid(
    transaction_freq=["W"],
    val_size=[0.2, 0.1],
    val_precision=[0.9],
    min_template_support=[0.0, 0.001],
    min_support=[0.01, 0.025, 0.05],
    min_confidence=[0.7, 0.8, 0.9],
)
for i, grid_entry in enumerate(tqdm(grid)):
    print(f"Starting {i+1}/{grid_len} ({(i+1)/grid_len:.1%})")
    model = AssociationRulesTemplatePredictor(
        min_support=grid_entry.min_support,
        min_confidence=grid_entry.min_confidence,
        min_template_support=grid_entry.min_template_support,
        val_size=grid_entry.val_size,
        val_precision=grid_entry.val_precision,
        transaction_freq=grid_entry.transaction_freq,
    )
    run_id = "ar_template_{}".format("_".join(str(var) for var in grid_entry))
    framework = TrainAndPredictFramework(
        model, group_key=["infobox_key", "property_name"], run_id=run_id
    )
    framework.load_data(input_path, n_files, n_jobs)
    framework.fit_model()
    framework.test_model()
    framework.generate_plots()
    del model, framework
