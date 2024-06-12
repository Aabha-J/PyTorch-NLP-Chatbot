from data_processing import get_json_data, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

BATCH_SIZE = 8
WORKERS = 2
HIDDEN_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 1000

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

    return X_train, Y_train, tags

class ChatDataset(Dataset):
    def __init__(self, x, y):
        self.n_samples = len(x)
        self.x_data = x
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
    
def train(model):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  for epoch in range(EPOCHS):
      for (words, labels) in train_loader:
          words = words.to(device)
          labels = labels.to(dtype=torch.long).to(device)

          #forward pass
          outputs = model(words)
          loss = criterion(outputs, labels)

          #backward and optimizer step
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      if (epoch + 1) % 100 == 0:
          print(f'epoch {epoch + 1}/{EPOCHS}, loss={loss.item():.3f}')

  print(f'Final Loss, loss={loss.item():.3f}')



if __name__ == "__main__":
    X_train, Y_train, tags = get_training_data()

    dataset = ChatDataset(x=X_train, y=Y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    model = NeuralNet(input_size=len(X_train[0]), hidden_size=HIDDEN_SIZE, num_classes=len(tags))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train(model)
    all_words, tags, xy = get_json_data()

    data = {
        "model_state": model.state_dict(),
        "input_size": len(X_train[0]),
        "hidden_size": HIDDEN_SIZE,
        "output_size": len(tags),
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)
    print("Done")
    
