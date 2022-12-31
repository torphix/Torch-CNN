import sys
import argparse
from src.trainer import Trainer
from src.utils import open_config
from src.text.data import TextPreprocessor
from src.text.model import load_model as text_load_model
from src.image.model import load_model as image_load_model
from src.text.data import load_dataset as text_load_dataset
from src.image.data import load_dataset as image_load_dataset


if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()

    if command == 'train':
        parser.add_argument('-t', '--type', type=str,
                            default='image', choices=['image', 'text'])
        parser.add_argument('-m', '--model', type=str, default='cifar10')
        args, lf_args = parser.parse_known_args()

        if args.type == 'image':
            config = open_config(args.model)
            train_dataloader, test_dataloader, classes = image_load_dataset(
                **config['data'])
            model_name = config['model'].pop('model_name')
            model = image_load_model(model_name, len(classes), config['model'])

            trainer = Trainer(
                config, model, train_dataloader, test_dataloader)
            trainer()

        elif args.type == 'text':
            config = open_config(args.model)
            # Preprocess Data
            processor = TextPreprocessor(**config['preprocessing'])
            processor()
            # Load data
            train_dataloader, test_dataloader, classes = text_load_dataset(
                **config['data'])
            model_name = config['model'].pop('model_name')
            model = text_load_model(model_name, len(classes), **config['model'])
            # Train
            trainer = Trainer(
                config, model, train_dataloader, test_dataloader)
            trainer()

    else:
        raise NotImplementedError(f'Command: {command} does not exist')
