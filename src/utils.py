import yaml


def open_config(name):
    with open(f'configs/{name}.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config
