from dataset_kaggle import GetKaggle
from torch.utils.data import DataLoader, sampler
import numpy as np
import torch
from rnn_test import RNN

batch_size = 64
validation_split = .2
shuffle_dataset = True
random_seed = 42
num_epochs = 10
num_classes = 5
num_layers = 4
hidden_size = 250
learning_rate = 0.0001
input_size = 400
sequence_length = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = GetKaggle('mustfkeskin/turkish-movie-sentiment-analysis-dataset',2)
#data_loader = DataLoader(data_bpe,shuffle=True,batch_size=batch_size,num_workers=4)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = sampler.SubsetRandomSampler(train_indices)
valid_sampler = sampler.SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

model = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length, device).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for e in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        scores = model(data)
        loss = criterion(scores, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_accuracy(loader, model):

    if loader.data.train:
        print('Train accuracy')
    else:
        print("test accuracy")
    num_correct = 0
    num_sample = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            scores = model(x.to(device))
            _, prediction = scores.max(1)
            num_correct += (prediction ==y.to(device)).sum()
            num_sample += prediction.size(0)

            model.train()


check_accuracy(train_loader, model)
check_accuracy(validation_loader, model)




