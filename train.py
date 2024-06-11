from data_processing import get_json_data, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 8
WORKERS = 2

def get_training_data():
    all_words, tags, xy = get_json_data()

    X_train = []
    Y_train = []

    for (pattern, tag) in xy:
        bag = bag_of_words(pattern, all_words)
        X_train.append(bag)

        label = tags.index(tag)
        Y_train.append(label)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train

class ChatDataset(Dataset):
    def __init__(self, X, Y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = Y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
def train():
    X_train, Y_train = get_training_data()

    dataset = ChatDataset(X=X_train, Y=Y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)


if __name__ == "__main__":
    train()
    print("Done")
    
