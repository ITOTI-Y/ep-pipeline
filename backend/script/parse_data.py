
from pickle import load

from backend.utils.config import ConfigManager


def parse_results_to_csv(config: ConfigManager):
    import pandas as pd

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
    for optimization_file in optimization_dir.glob("**/result.pkl"):
        with open(optimization_file, "rb") as f:
            result = load(f)
            print(result)
