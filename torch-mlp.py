import torch
from torch.nn import Softmax, MSELoss, Linear, CrossEntropyLoss, Dropout
from torch import nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, RandomSampler, random_split
from cv2 import imread, IMREAD_GRAYSCALE
import os

epochs = 5
lr = 0.01
data_len = 42000
train_split_percent = 0.8
test_split_percent = 1 - train_split_percent
dropout_prob = 0.2
            
class NN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input = Linear(784, 500)
        self.hidden = Linear(500, 250)
        self.output = Linear(250, 10)
        self.dropout = Dropout(dropout_prob)
        self.softmax = Softmax(dim=0)
        self.loss_fn = CrossEntropyLoss()
        self.opt = SGD(self.parameters(), lr=lr)
        self.lr = lr
    
    def forward(self, data):
        img, label = data
        img = torch.flatten(img)
        img = img.div(255)

        input = F.relu(self.input.forward(img))
        input = self.dropout(input)
        hidden = F.relu(self.hidden.forward(input))
        hidden = self.dropout(hidden)
        output = self.output.forward(hidden)

        res = self.softmax(output)
        loss_label = torch.zeros(10)
        loss_label.data[label] = 1
        loss = self.loss_fn(output, loss_label)
        # print("L:", loss)
        pred = torch.argmax(res)
        return loss, pred == label

    def backward(self, loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

class TrainData(Dataset):
    def __init__(self, data_pth) -> None:
        super().__init__()
        self.data = {}
        k = 0
        for i in range(0, 10):
            for filename in os.listdir(f"{data_pth}/{i}"):
                if filename is not None:
                    img = imread(f"{data_pth}/{i}/{filename}", IMREAD_GRAYSCALE)
                    if img is not None:
                        img_data = torch.tensor(torch.from_numpy(img), dtype=torch.float)
                        self.data[k] = (img_data, i)
                        self.data[k][0].requires_grad = True
                        k += 1

    def __getitem__(self, index):
        return self.data[index]

tot_train_loss = 0
tot_test_loss = 0
train_accuracy = []
test_accuracy = []
id = 0

network = NN()

data = TrainData("trainingSet")

train, test = random_split(data.data, [train_split_percent, test_split_percent])

for epoch in range(epochs):
    print(f"Epoch: {epoch+1}/{epochs}")
    for i, item in enumerate(train):
        id += 1
        loss, good = network.forward(item)
        network.backward(loss)
        tot_train_loss += loss
        train_accuracy.append(good)
        if i % (data_len*train_split_percent/20) == 0:
            print("curr loss: ", tot_train_loss/id)
    
    print(f"Avg loss: ", tot_train_loss/((epoch+1)*(data_len*train_split_percent)))
    print(f"Accuracy: {sum(train_accuracy)/len(train_accuracy)}")
    train_accuracy = []

for item in test:
    loss, good = network.forward(item)
    tot_test_loss += loss
    test_accuracy.append(good)

print(f"Eval loss: ", tot_test_loss/data_len*test_split_percent)
print(f"Accuracy: {sum(test_accuracy)/len(test_accuracy)}")
