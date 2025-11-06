import yaml
from easydict import EasyDict as edict


def load_config(file):
    """Load configuration from YAML file.

    Parameters
    ----------
    file : str
        Path to configuration file.

    Returns
    -------
    conf_dict : easydict.EasyDict
        Dict containing configurations.

    """
    # read configuration files
    with open(file, "r") as f:
        conf_dict = edict(yaml.safe_load(f))
    return conf_dict
