import torch
import os
from cv2 import imread, IMREAD_GRAYSCALE
import csv


class Dataset():
    '''
    Generic Dataset class.

    Includes getter, split/shuffle, and generator.
    '''
    def __init__(self, data) -> None:
        self.data = data
        self.len = len(data)

    def __getitem__(self, index) -> tuple:
        return self.data[index]
    
    '''
    Randomly shuffles and splits dataset into two separate sets.

    The split must be >=1. set_2 := split, set_2 := 1 - split.

    Returns a tuple of dicts.
    '''
    def split_shuffle(self, split) -> tuple:

        assert(split <= 1)

        set_1, set_2 = {}, {}
        idx_1, idx_2 = 0, 0
        for i in torch.randperm(self.len):
            if (idx_1 < (self.len * split)):
                set_1[idx_1] = self.data[i.item()]
                idx_1 += 1
            else:
                set_2[idx_2] = self.data[i.item()]
                idx_2 += 1

        return set_1, set_2

    def data_generator(self):
        for i in range(self.len):
            yield self.data[i]

class TitanicDataset(Dataset):
    '''
    Data class for titanic data, takes in csv files test.csv and train.csv.

    Contains 10 features + whether a passenger survived for train.csv. I am only
    loading 7 features.
    '''
    def __init__(self, data_pth) -> None:
        self.data = {}
        with open(f"{data_pth}/train.csv", newline='\n') as f:
            reader = csv.DictReader(f)
            k = 0
            for row in reader:
                row_info = [
                            float(row['PassengerId']),
                            float(row['Pclass']),
                            float(row['Sex'] == 'male'),
                            float(row['Age']) if row['Age'] != '' else 0, # Handling of missing age
                            float(row['SibSp']),
                            float(row['Parch']),
                            float(row['Fare'])
                            ]

                self.data[k] = (torch.tensor(row_info, dtype=torch.float), int(row['Survived']))
                k += 1

        self.len = len(self.data)

class MNISTDataset(Dataset):
    '''
    Data class for MNIST data, takes in a data_pth and processes sub-folders,
    or takes in a dictionary of data.

    MNIST images are 28x28 grayscale.

    If loaded from data_pth each image is stored as a tuple with the data and the label.
'''
    def __init__(self, data=None, data_pth=None) -> None:
        if data != None:
            assert(type(data) is dict)
            self.data = data
            self.len = len(data)
        else: 
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
            self.len = k