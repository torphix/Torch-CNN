import sys
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.model import VanillaCNN
from prettytable import PrettyTable
from src.data import load_canoncial_vision_datasets


class Trainer():
    def __init__(self, config):
        with open(config, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        # Init hyperparams
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = config['epochs']
        self.log_steps = config['log_n_steps']
        self.log_run = config['log_run']
        self.use_early_stopping = config['use_early_stopping']
        self.early_stopping_patience = config['early_stopping_patience']

        # Init Modules
        self.model = VanillaCNN(**config['model']).to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(
            self.model.parameters(), **config['optim'])

        # Load Data
        self.train_dataloader, self.val_dataloader, classes = load_canoncial_vision_datasets(
            **config['data'])

        # Init Log Metrics
        self.metrics_table = PrettyTable(
            ['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])
        self.running_val_acc = []
        self.running_val_loss = []
        self.running_train_acc = []
        self.running_train_loss = []

    def __call__(self):
        # Init logs
        if self.log_run:
            print(self.metrics_table)

        # Start Training
        for e in range(self.epochs):
            train_loss, train_acc = self.train_iter(self.train_dataloader, e)
            val_loss, val_acc = self.eval_iter(self.val_dataloader, e)

            if self.use_early_stopping:
                self.check_early_stopping(val_acc)

            if e % self.log_steps == 0:
                self.log_metrics(e,
                                 train_loss,
                                 train_acc,
                                 val_loss,
                                 val_acc)

        final_outputs = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        self.vis_results()
        return final_outputs

    def train_iter(self, dataloader, epoch):
        running_loss, running_acc = 0, 0
        for x, y in tqdm(dataloader, f'Train Epoch: {epoch}', leave=False):
            x, y = x.to(self.device), y.to(self.device)
            # Forward Pass
            outputs = self.model(x)
            loss = self.loss(outputs, y)
            # Optimization
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            running_acc += int(outputs.argmax() == y.argmax())
            running_loss += loss.detach().cpu().item()

        running_acc /= len(dataloader)
        running_loss /= len(dataloader)
        return running_loss, running_acc

    def eval_iter(self, dataloader, epoch):
        running_loss, running_acc = 0, 0
        for x, y in tqdm(dataloader, f'Val Epoch: {epoch}', leave=False):
            x, y = x.to(self.device), y.to(self.device)
            # Forward Pass
            outputs = self.model(x)
            loss = self.loss(outputs, y)

            running_acc += int(outputs.argmax() == y.argmax())
            running_loss += loss.detach().cpu().item()

        running_acc /= len(dataloader)
        running_loss /= len(dataloader)
        return running_loss, running_acc

    def vis_results(self):
        fig, axs = plt.subplots(2)
        fig.suptitle('Training Results')
        # Loss Plot
        epochs = [i for i in range(len(self.running_train_loss))]
        axs[0].plot(epochs, self.running_train_loss)
        axs[0].plot(epochs, self.running_val_loss)
        axs[0].set_xlabel('Epochs')
        axs[0].set_title('Training Loss')
        # Accuracy Plot
        axs[1].plot(epochs, [i*100 for i in self.running_train_acc])
        axs[1].plot(epochs, [i*100 for i in self.running_val_acc])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Training Accuracy')
        axs[1].set_ylim(0, 100)
        plt.show()

    def check_early_stopping(self, val_acc):
        if self.use_early_stopping:
            running_accs = self.running_val_acc[-self.early_stopping_patience:]
            val_check = sum(
                [1 if val_acc >= acc else 0 for acc in running_accs])
            if val_check == 0:
                test_loss, test_acc = self.eval_iter(self.train_dataloader)
                print(f'Final Testset Accuracy {test_acc}')
                sys.exit(
                    f'Val Accuracy has not increased for {self.early_stopping_patience} epochs, Stopping')

    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.running_val_acc.append(val_acc)
        self.running_val_loss.append(val_loss)
        self.running_train_acc.append(train_acc)
        self.running_train_loss.append(train_loss)

        self.metrics_table.add_row([
            epoch,
            f'{round(train_loss, 4)}',
            f'{round(train_acc, 2)}%',
            f'{round(val_loss, 4)}',
            f'{round(val_acc, 2)}%'])
        if self.log_run:
            # Print only new row
            print("\n".join(self.metrics_table.get_string().splitlines()[-2:]))
