import yaml

from directories import APP_DIR


def load_config_file():
    filepath = APP_DIR / "config.yaml"
    config = load_yaml_file(filepath)
    return config


def load_yaml_file(filepath):
    with open(filepath, encoding="utf8") as file:
        obj = yaml.load(file, Loader=yaml.loader.SafeLoader)
    return obj
