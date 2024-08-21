import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import numpy
from dataset import Dataset

sd = 42

torch.manual_seed(sd)
numpy.random.seed(sd)
random.seed(sd)

class Linear():
    '''
    Individual linear layer for a network. Takes in number of input and output features.
    '''
    def __init__(self, in_features, out_features) -> None:
        self.weights = torch.empty((out_features, in_features), requires_grad=True)

        # Initializing weights based on Kaiming/He Weight Initialization, since Xavier Weight Initialization
        # apparently can have bad effect on networks that use non-linear activation functions like ReLU
        stddev = np.sqrt(2. / in_features)
        self.weights.data.normal_(0, stddev)

    def __call__(self, input) -> Tensor:
        return self.forward(input)

    def forward(self, input) -> Tensor:
        return torch.matmul(input, self.weights.t())

    '''
    Updates the weights of a layer according to gradient descent optimization.
    '''
    def update_weights(self, lr) -> None:
        self.weights.data.sub_(self.weights.grad, alpha=lr) # Subtraction done in place to maintain gradients
        self.weights.grad.zero_()

class ReLU():
    '''
    Applies Rectified Linear Unit activation function inplace.

    Returns 0 if negative and the number if positive.

    f(x) = max{0, x}
    '''
    def __init__(self) -> None:
        pass

    def __call__(self, input) -> Tensor:
        return self.forward(input)
    
    def forward(self, input) -> Tensor:
        for i, x in enumerate(input.data):
            if x > 0:
                input.data[i] = x
            else:
                input.data[i] = 0
        return input

class Dropout():
    '''
    Applies dropout function. Randomly drops out p% of neuron activations by setting
    them to zero.

    This function helps reduce overfitting.
    '''
    def __init__(self, p=0.2) -> None:
        if p < 0.0 or p > 1.0:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.dropout_prob = p

    def __call__(self, input, eval=False) -> Tensor:
        return self.forward(input, eval)

    def forward(self, input, eval=False) -> Tensor:
        if eval or self.dropout_prob == 0:
            return input
        dropped_num = input.shape[0] * self.dropout_prob
        dropped = torch.randperm(input.shape[0])[:int(dropped_num)]
        for drop in dropped:
            input.data[drop] = 0
        return input

class MLP():
    '''
    Basic multi-layer perceptron with one hidden layer, ReLU activation functions, dropout and
    CrossEntropyLoss.

    CrossEntropyLoss() combines nn.LogSoftmax() and nn.NLLLoss() in one single class, so no
    softmax is needed, however I still use softmax to get the actual prediction. I decided to
    write my own Linear and ReLU layers, but for ease of backpropogation decided to stick with
    the built-in PyTorch Softmax and CEL.

    In the future I will implement all my own functions, including backpropagation calculations.

    To initialize with layers create an object of the class with:

        layers = [
            (in_features, out_features),
            ...
        ]

    for as many layers as wanted. ReLU and Dropout are applied to all except last output layer.
    '''
    def __init__(self, layers, dp=0.2) -> None:
        self.layers = []
        for layer in layers:
            self.layers.append(Linear(*layer))
        self.relu = ReLU()
        self.dropout = Dropout(p=dp)
        self.softmax = torch.nn.Softmax(dim=0)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = 0.01
        self.eval_ = False

    def eval(self):
        self.eval_ = True

    def save_weights(self, filename) -> None:
        weights = []
        
        for layer in self.layers:
            weights.append(torch.clone(layer.weights))

        torch.save(weights, filename)
        print(f"{filename} saved.")


    def load_weights(self, filename) -> None:
        weights = torch.load(filename, weights_only=True)

        for i, layer in enumerate(self.layers):
            layer.weights = weights[i]
        
        print(f"{filename} loaded.")

    '''
    This function is cool and can be used to visualize the first layers gradients. 
    You can uncomment the function in backward to see.
    '''
    def visualize_gradient(self, gradient) -> None:
        f, axarr = plt.subplots(1, 2)

        # Need to transpose to reverse shape so it is output x input to select one input neuron
        axarr[0].imshow(gradient[0].view(1, 5))
        axarr[1].imshow(gradient[0].view(1, 5) == 0)
        plt.show()

    def backward(self, loss) -> None:
        loss.backward()
        # self.visualize_gradient(self.layers[0].weights.grad)
        for layer in self.layers:
            layer.update_weights(self.lr)

    '''
    Forward pass through model. ReLU and dropout are applied to first two layer's
    outputs.

    Takes in a tuple with tensor and data label.

    Returns loss and whether the model predicted the right label.
    '''
    def forward(self, input):
        data, label = input
        data = torch.flatten(data) # flatten if not already flat
        data = data.div(max(data)) # normalize pixel values between 0 and 1
        x = data
        
        # Run through network, applying relu and dropout to all layers except output
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = self.dropout(self.relu(layer(x)), eval=self.eval_)

        output = self.layers[len(self.layers)-1](x)

        # Create one-hot encoding truth vector and calc loss
        truth_label = torch.zeros(output.shape[0])
        truth_label.data[label] = 1
        loss = self.loss_fn(output, truth_label)

        # print("L:", loss)
        res = self.softmax(output)
        pred = torch.argmax(res)
        # print(pred, label)

        if torch.isnan(loss):
            raise ValueError(f"NaN detected, loss: {loss}, pred: {pred}, truth: {label}")

        return loss, pred == label

    def train(self, data, epochs, lr, train_val_split=0.8) -> None:
        self.eval_ = False
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.lr = lr
        if epochs < 0:
            raise ValueError(f"Invalid epochs: {epochs}")

        id = 1
        tot_train_loss = 0

        train_data, val_data = Dataset(data).split_shuffle(train_val_split) # split again to get validation set

        for epoch in range(epochs):
            train_accuracy = []

            print("---------------Training---------------")
            print(f"Epoch: {epoch+1}/{epochs}")
            t = tqdm(enumerate(range(len(train_data))), total=len(train_data), desc=f"Loss: {(tot_train_loss/id):0.4f}")
            for i, idx in t:
                loss, good = self.forward(train_data[idx])
                self.backward(loss)

                tot_train_loss += loss.item()
                train_accuracy.append(good) # track whether pred was good

                if i % (len(train_data)/20) == 0:
                    t.set_description(f"Loss: {(tot_train_loss/id):0.4f}")
                id += 1
            
            print(f"Avg Training Loss: ", tot_train_loss/((epoch+1)*(len(train_data))))
            print(f"Training Accuracy: {sum(train_accuracy)/len(train_accuracy)}")
            print("--------------------------------------")
            self.val(val_data)

    def val(self, data) -> None:
        self.eval_ = True
        tot_val_loss = 0
        val_accuracy = []

        print("-----------------Eval-----------------")
        with torch.no_grad():
            for _, idx in tqdm(enumerate(range(len(data))), total=len(data)):
                loss, good = self.forward(data[idx])

                tot_val_loss += loss.item()
                val_accuracy.append(good)
                
            print(f"Validation Loss: ", tot_val_loss/len(data))
            print(f"Validation Accuracy: {sum(val_accuracy)/len(val_accuracy)}")
        print("--------------------------------------")
        
        self.eval_ = False