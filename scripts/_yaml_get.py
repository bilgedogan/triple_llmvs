"""Tiny CLI helper: read a key from a YAML file. Used by the .sh wrappers
to extract `splits`, `cuda_devices`, `pretrained_aggregator`, etc.

Usage:
    python scripts/_yaml_get.py <yaml_path> <dotted.key> [--default <val>]

Lists are emitted as space-separated for shell `for` loops.
"""
import argparse
import sys

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('key')
    ap.add_argument('--default', default='')
    args = ap.parse_args()

    with open(args.path, 'r') as f:
        data = yaml.safe_load(f) or {}

    cur = data
    for part in args.key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            print(args.default)
            return
        cur = cur[part]

    if isinstance(cur, list):
        print(' '.join(str(x) for x in cur))
    elif isinstance(cur, bool):
        print('true' if cur else 'false')
    elif cur is None:
        print(args.default)
    else:
        print(cur)


if __name__ == '__main__':
    main()
