import torch
import pickle
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



class TextPreprocessor():
    def __init__(self, file_path: str, 
                train_test_split: list, 
                n_samples_per_class: int, 
                remove_classes: list = [],  
                vocab_black_list:list=[], 
                output_path:str = './data'):
        self.file_path = file_path
        self.train_test_split = train_test_split
        self.n_samples_per_class = n_samples_per_class
        self.remove_classes = remove_classes
        self.vocab_blacklist = vocab_black_list
        self.output_path = output_path

    def __call__(self):
        df = self.sample_data()
        vocab = self.construct_vocab(df)
        train_df, test_df = self.split_dataset(df)
        # Save data
        train_df.to_csv(f'{self.output_path}/train.csv')
        test_df.to_csv(f'{self.output_path}/test.csv')
        # Save Vocab
        with open(f'{self.output_path}/vocab.pickle', 'wb') as handle:
            pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Save Classes
        classes = {c:i for i, c in enumerate(train_df['class'].unique())}
        with open(f'{self.output_path}/classes.pickle', 'wb') as handle:
            pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return train_df, test_df
    
    def sample_data(self):
        '''
        - Oversamples smaller classes and undersamples large classes to reach n_samples_per_class
        - Removes unwanted classes
        1) Class balance
        2) Create a dictionary from remaining text
        3) Split csv into train test
        '''
        df = pd.read_csv(self.file_path)

        class_freq = df['class'].value_counts()
        classes = df['class'].unique()
        dfs = []
        for c in classes:
            # Balance classes (oversampling & undersampling) + Remove unwanted classes
            if c not in self.remove_classes:
                dfs.append(df[df['class'] == c].sample(
                    self.n_samples_per_class,
                    replace=False if len(df) < self.n_samples_per_class else True))
        df = pd.concat(dfs)
        return df

    def construct_vocab(self, df:pd.DataFrame):
        vocab = {'<pad>':0}
        for i, row in df.iterrows():
            for c in row['text']:
                if c not in self.vocab_blacklist:
                    vocab[c] = len(vocab.keys())
        return vocab

    def split_dataset(self, df:pd.DataFrame):
        train_df, test_df = train_test_split(df, test_size=self.train_test_split[1], shuffle=True, stratify=df['class'])
        return train_df, test_df


class TextClassificationDataset(Dataset):
    def __init__(self, csv_path:str, vocab_path:str, classes_pickle_path:str):
        '''
        Expects a csv file with columns: class and text 
        and a pickled vocab to tokenize the text
        '''
        super().__init__()
        self.data = pd.read_csv(csv_path)

        with open(vocab_path, 'rb') as handle:
            self.vocab = pickle.load(handle)
        with open(classes_pickle_path, 'rb') as handle:
            self.class_dict = pickle.load(handle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_text = self.data['text'][idx]
        text = torch.tensor([self.vocab[c] for c in raw_text])
        label = self.class_dict[self.data['class'][idx]]
        # label_vec = torch.zeros((len(self.class_dict)))
        # label_vec[label] = 1
        # return text, label_vec.long()
        return text, label

    def collate_fn(self, batch):
        max_len = max([text.shape[0] for text, _ in batch])
        padded_text, labels = [], []
        for text, label in batch:
            padded_text.append(
                F.pad(text, (max_len-text.shape[0], 0), value=0))
            labels.append(torch.tensor(label))
        return torch.stack(padded_text), torch.stack(labels)
            

def load_dataset(train_csv, test_csv, vocab_path, classes_path, batch_size=16, num_workers=4):

    train_ds = TextClassificationDataset(train_csv, vocab_path, classes_path)
    test_ds = TextClassificationDataset(test_csv, vocab_path, classes_path)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=train_ds.collate_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=train_ds.collate_fn)
    return train_dl, test_dl, train_ds.class_dict
    