import copy
from typing import Any

import hydra
import optuna
import rootutils
from omegaconf import DictConfig, OmegaConf, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# see `src/train.py` for what `setup_root` does
# ------------------------------------------------------------------------------------ #

from src.train import train
from src.utils import RankedLogger, extras, get_metric_value

log = RankedLogger(__name__, rank_zero_only=True)


def suggest(trial: optuna.Trial, name: str, spec: DictConfig) -> Any:
    """Sample a hyperparameter value from its search space spec.

    Supported specs:
        - `{key: ..., type: float, low: 1e-5, high: 1e-2, log: true}`
        - `{key: ..., type: int, low: 1, high: 8, step: 1, log: false}`
        - `{key: ..., type: categorical, choices: [32, 64, 128]}`

    :param trial: The Optuna trial.
    :param name: The parameter name.
    :param spec: The search space spec.
    :return: The sampled value.
    """
    kind = spec.get("type")
    if kind == "float":
        return trial.suggest_float(
            name, spec.low, spec.high, step=spec.get("step"), log=spec.get("log", False)
        )
    if kind == "int":
        return trial.suggest_int(
            name, spec.low, spec.high, step=spec.get("step", 1), log=spec.get("log", False)
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, OmegaConf.to_container(spec.choices))
    raise ValueError(f"Unknown search space type <{kind}> for <{name}>!")


@hydra.main(version_base="1.3", config_path="../configs", config_name="tune.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for hyperparameter search with Optuna.

    Every trial overrides the config `key` of each entry in `cfg.optuna.params` with a sampled
    value, runs `train()` in its own `<output_dir>/trials/<number>` directory and returns the
    value of `cfg.optimized_metric`.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: The best value of the optimized metric.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    tune_cfg = cfg.optuna
    output_dir = cfg.paths.output_dir

    study = optuna.create_study(
        study_name=tune_cfg.get("study_name"),
        storage=tune_cfg.get("storage"),
        direction=tune_cfg.direction,
        sampler=hydra.utils.instantiate(tune_cfg.sampler) if tune_cfg.get("sampler") else None,
        pruner=hydra.utils.instantiate(tune_cfg.pruner) if tune_cfg.get("pruner") else None,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = copy.deepcopy(cfg)
        with open_dict(trial_cfg):
            for name, spec in tune_cfg.params.items():
                OmegaConf.update(trial_cfg, spec.key, suggest(trial, name, spec), merge=False)
            trial_cfg.paths.output_dir = f"{output_dir}/trials/{trial.number}"

        log.info(f"Trial {trial.number}: {trial.params}")
        metric_dict, _ = train(trial_cfg)
        return get_metric_value(metric_dict=metric_dict, metric_name=cfg.optimized_metric)

    study.optimize(objective, n_trials=tune_cfg.n_trials, timeout=tune_cfg.get("timeout"))

    log.info(f"Best trial: {study.best_trial.number}")
    log.info(f"Best value: <{cfg.optimized_metric}={study.best_value}>")
    # save best values as `<config key>: <value>`, ready to be used as overrides
    best = {tune_cfg.params[name].key: value for name, value in study.best_params.items()}
    log.info(f"Best params: {best}")
    with open(f"{output_dir}/best_params.yaml", "w") as file:
        file.write(OmegaConf.to_yaml(OmegaConf.create(best)))

    return study.best_value


if __name__ == "__main__":
    main()
