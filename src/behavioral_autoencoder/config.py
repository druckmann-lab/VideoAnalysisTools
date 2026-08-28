'''
Single definition of the experiment-config merge.

Lives in the package rather than in a training script because several entry
points need it -- training, inference and the epoch-time benchmark must agree on
exactly which config a checkpoint was produced under. A second copy of this
logic is how they stop agreeing: the inference script carried its own for a
while and silently predated "extends", so it resolved a variant env to base
defaults instead of the config the checkpoint was actually trained with.

2026.08.28. Balint w/ Claude
'''

import json
import os

REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_DIR = os.path.join(REPO_ROOT, 'configs')
BASE_CONFIG = 'ae_config.json'


def update_dict(d, u):
    """Recursively updates a nested dictionary."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def strip_private(d):
    """
    Drops "_"-prefixed keys recursively.

    JSON has no comments, so env configs carry "_comment" instead; without this
    they would survive the merge into the config.json saved beside every
    checkpoint.
    """
    return {k: (strip_private(v) if isinstance(v, dict) else v)
            for k, v in d.items() if not k.startswith('_')}


def available_envs(config_dir=None):
    config_dir = config_dir or CONFIG_DIR
    return sorted(f[: -len('_config.json')] for f in os.listdir(config_dir)
                  if f.endswith('_config.json') and f != BASE_CONFIG)


def load_config(env_name, config_dir=None):
    """
    Loads the base config and overwrites it with environment specifics.

    An env config may carry "extends": "<other_env>" to inherit from another env
    config before its own keys are applied. This exists so a variant -- e.g. one
    arm of a scheduler A/B -- can be a two-line delta rather than a full copy of
    its parent. Duplicated config files drift, and a drifted arm silently
    invalidates the comparison it was built for.

    Raises rather than warning on a missing env config: falling back to the base
    config silently yields batch 32, 50 epochs and no checkpoint_dir -- a wrong
    experiment that looks like a real one until it fails much later.
    """
    config_dir = config_dir or CONFIG_DIR

    with open(os.path.join(config_dir, BASE_CONFIG), 'r') as f:
        config = json.load(f)

    chain, name, seen = [], env_name, set()
    while name:
        env_path = os.path.join(config_dir, f'{name}_config.json')
        if not os.path.exists(env_path):
            raise FileNotFoundError(
                f"env config not found: {env_path}\n"
                f"available envs: {', '.join(available_envs(config_dir))}")
        if name in seen:
            raise ValueError(f"circular 'extends' in config chain at {name!r}")
        seen.add(name)
        with open(env_path, 'r') as f:
            env_config = json.load(f)
        # popped so it does not survive into the saved config.json
        name = env_config.pop('extends', None)
        chain.append(strip_private(env_config))

    # Parents first, so a child's keys win over the ones it inherits.
    for env_config in reversed(chain):
        config = update_dict(config, env_config)

    return config
