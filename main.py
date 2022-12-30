import sys
import yaml
import argparse
from src.model import load_model
from src.utils import open_config
from src.data import load_dataset
from src.train import MultiClassTrainer


if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()

    if command == 'train':

        parser.add_argument('-m', '--model', type=str, default='cifar10')
        args, lf_args = parser.parse_known_args()

        config = open_config(args.model)
        train_dataloader, test_dataloader, classes = load_dataset(
            **config['data'])
        model_name = config['model'].pop('model_name')
        model = load_model(model_name, len(classes), config['model'])

        trainer = MultiClassTrainer(
            config, model, train_dataloader, test_dataloader)
        trainer()
