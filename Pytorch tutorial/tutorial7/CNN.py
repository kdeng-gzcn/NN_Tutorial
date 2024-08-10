import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import DogsVSCats

# a trick
REBUILD_DATA = False

if REBUILD_DATA: # never run this again!
    dogsvscats = DogsVSCats()
    dogsvscats.make_training_data()

training_data = np.load('training_data.npy', allow_pickle=True) # ???

# reshape our data in nn
X = torch.Tensor(np.array([i[0] for i in training_data])).view(-1, 50, 50)
X = X / 255 # each pixel is in range [0, 255], this line is to scale the data

y = torch.Tensor(np.array([i[1] for i in training_data]))

valid_perc = 0.1
valid_size = int(len(X) * valid_perc)
valid_size

train_X = X[:-valid_size]
train_y = y[:-valid_size]

test_X = X[-valid_size:]
test_y = y[-valid_size:]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define conv layer
        self.conv1 = nn.Conv2d(
            in_channels=1, # our image is 1 * 50 *50
            out_channels=32,
            kernel_size=5
        )
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # define pool layer
        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        self.pool2 = nn.MaxPool2d((2, 2))
        self.pool3 = nn.MaxPool2d(2)

        # define fc layer, but need to know the shape
        value = 512

        self.fc1 = nn.Linear(
            in_features=value, 
            out_features=512
        )
        
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):

        # a standard CNN structure is input -> (conv layer -> active func -> pool layer) -> next structure 
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        # for first time, make sure the dim of x
        x = x.flatten(start_dim=1)
        # print(x.shape)

        # after feature extraction, use fc to do classification
        # a standard FC structure is input -> (fc layer -> active func) -> next structure
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1) # make sure each row is a distribution (dim=1)

        return x
    

def train(net):
    optimizer = optim.Adam(net.parameters(), lr=0.001) # optimizer is supervising the net.params on GPU
    loss_function = nn.MSELoss()

    batch_size = 128
    epochs = 3

    # not using the API from torch.utils.data.DataLoader

    for epoch in range(epochs):
        for i in tqdm(range(0, len(train_X), batch_size)):
            # print(i, i + batch_size)
            batch_X = train_X[i:i + batch_size].view(-1, 1, 50, 50).to(device) # claim the channel number
            batch_y = train_y[i:i + batch_size].to(device) # no need to reshape?

            # make sure the optimizer or net
            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1} loss: ', loss.item())

def test(net):
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(test_X))): # batch = 1
            y_ = torch.argmax(test_y[i]).to(device)
            prob_hat = net(test_X[i].view([-1, 1, 50, 50]).to(device))[0] # reshape it into dim1
            y_hat = torch.argmax(prob_hat)

            if y_hat == y_:
                correct += 1
            
            total += 1
        
        print('Accuracy: ', round(correct / total, 3))


print(torch.cuda.is_available())
device = torch.device('cuda:0')
print(torch.cuda.device_count())

net = Net()
net.to(device)

train(net)
test(net)
