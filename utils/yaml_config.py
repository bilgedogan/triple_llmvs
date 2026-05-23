"""YAML config support: merge YAML overrides into an argparse Namespace.

Precedence (highest to lowest):
    1. Explicit CLI flags (anything the user typed on the command line).
    2. YAML file values (when --config is given).
    3. argparse defaults.

The driver scripts (pretrain_aggregator.py / train_rl.py / test_rl.py) call
``apply_yaml(parser, opt)`` after parsing args.
"""
import sys

import yaml


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def _cli_provided_keys(parser):
    """Return set of dest names the user explicitly passed on the CLI."""
    cli = set()
    argv = sys.argv[1:]
    for action in parser._actions:
        if not action.option_strings:
            continue
        if any(opt in argv for opt in action.option_strings):
            cli.add(action.dest)
    return cli


def apply_yaml(parser, opt, yaml_path):
    """Merge ``yaml_path`` into ``opt`` in place, respecting CLI overrides."""
    if yaml_path is None:
        return opt
    data = load_yaml(yaml_path)
    cli_keys = _cli_provided_keys(parser)
    for k, v in data.items():
        if not hasattr(opt, k):
            setattr(opt, k, v)
            continue
        if k in cli_keys:
            continue  # CLI wins
        setattr(opt, k, v)
    return opt
