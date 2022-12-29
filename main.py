import sys
import argparse
from src.train import Trainer


if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()

    if command == 'train':
        trainer = Trainer('config.yaml')
        trainer()