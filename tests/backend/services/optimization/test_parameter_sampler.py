from pathlib import Path

from backend.services.optimization.parameter_sampler import ParameterSampler
from backend.utils.config import ConfigManager

CONFIG = ConfigManager(Path("backend/configs"))


def _sample(seed: int) -> list[dict]:
    samples = ParameterSampler(CONFIG, seed=seed).sample(
        n_samples=64, building_type="coa"
    )
    return [sample.model_dump() for sample in samples]


def test_same_seed_gives_same_samples() -> None:
    assert _sample(seed=1) == _sample(seed=1)


def test_different_seeds_give_different_samples() -> None:
    assert _sample(seed=1) != _sample(seed=2)
