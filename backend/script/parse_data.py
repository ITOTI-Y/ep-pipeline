from pickle import load

import pandas as pd

from backend.utils.config import ConfigManager


def parse_results_to_csv(config: ConfigManager):
    results_dir = config.paths.ecm_dir

    results = []
    for result_file in results_dir.glob("**/result.pkl"):
        with open(result_file, "rb") as f:
            result = load(f)
            if result.ecm_parameters is None:
                continue
            code = {"code": result_file.parents[1].name}
            ecm_parameters = result.ecm_parameters.model_dump()
            eui_result = result.get_eui_summary()
            all_data = dict(sorted({**code, **ecm_parameters, **eui_result}.items()))
            results.append(all_data)
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "results.csv", index=False)


def parse_optimal_data(config: ConfigManager):
    optimization_dir = config.paths.optimization_dir
    baseline_dir = config.paths.baseline_dir
    optimization_files = list(optimization_dir.glob("**/result.pkl"))
    baseline_files = list(baseline_dir.glob("**/result.pkl"))
    optimization_files.sort()
    baseline_files.sort()
    for optimization_file, baseline_file in zip(
        optimization_files, baseline_files, strict=True
    ):
        with open(optimization_file, "rb") as f:
            optimization_result = load(f)
        with open(baseline_file, "rb") as f:
            baseline_result = load(f)
        print(optimization_result)
        print(baseline_result)


def parse_result_parameters(config: ConfigManager):
    import json

    optimization_dir = config.paths.optimization_dir
    optimization_files = list(optimization_dir.glob("**/result.pkl"))
    optimization_files.sort()
    for optimization_file in optimization_files:
        with open(optimization_file, "rb") as f:
            optimization_result = load(f)
        with open(optimization_file.with_suffix(".json"), "w") as f:
            json.dump(optimization_result.ecm_parameters.to_dict(), f, indent=4)
