import os
import yaml
import torch
import pprint
import shutil
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, config_path, overrides=None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print("Device:", self.device)

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # CLI overrides win over YAML values
        if overrides:
            for k, v in overrides.items():
                if v is not None:
                    cfg[k] = v

        for k, v in cfg.items():
            setattr(self, k, v)

        self._config_path = config_path
        self.set_dataset_dir()

    def set_dataset_dir(self):
        self.save_dir_root = f'Summaries/{self.experiment_name}/split_{self.split_idx}'
        os.makedirs(self.save_dir_root, exist_ok=True)

        # Copy original experiment YAML
        shutil.copy(self._config_path, os.path.join(self.save_dir_root, 'config.yaml'))

        # Save resolved config (with all overrides applied)
        resolved = {k: v for k, v in self.__dict__.items()
                    if not k.startswith('_') and k != 'device'}
        with open(os.path.join(self.save_dir_root, 'config_resolved.yaml'), 'w') as f:
            yaml.dump(resolved, f, default_flow_style=False, sort_keys=True)

    def __repr__(self):
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str
