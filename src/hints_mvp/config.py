"""hints_mvp config loading."""
import yaml

from .data import CHEXPERT, FINDINGS

CONDITIONS = ["no_hint", "no_hint_rerun", "own_trail", "other_trail", "random_trail", "leak_check"]


def load_config(path):
    cfg = yaml.safe_load(open(path))
    unknown = set(cfg["conditions"]) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions {unknown}")
    # The mock backend needs to know which findings to fill in; the other backends ignore this.
    cfg["labeler"]["findings"] = FINDINGS
    cfg["reader"]["findings"] = CHEXPERT
    return cfg
