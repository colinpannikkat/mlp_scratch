# MLP (for MNIST) from Scratch

## Overview
The goal of this project was to better understand basic neural networks. Since I am a very hands on learner, I thought the best way to solidify my understanding was build it from scratch! I decided to build a MLP (multi-layer perceptron) with one hidden layer to classify the [MNIST as JPG](https://www.kaggle.com/datasets/scolianni/mnistasjpg/data) dataset. Sadly, there were no labels provided for the test set, so I just used the training set as the whole dataset and created my own train/val/test splits.

Of course, I am still using the PyTorch library for tensor multiplication and other tensor math, and NumPy for basic square roots, but my goal is for everything to be written from scratch eventually, besides menial annoying low-level functions like square roots (unless I want to practice numerical approximation!). 

So far I have implemented:
* Simple linear/fully-connected layer
  * Includes weight optimization via gradient descent (using BGD or SGD would probably lead to greater accuracy)
* ReLU
* Dropout layer
* Simple dataloader for MNIST-JPG (using OpenCV for image loading)
  * Includes split function for creating train/test splits
* Weight saving and loading

I am still using the PyTorch implementation for:
* Softmax
* CrossEntropyLoss (This combines nn.LogSoftmax() and nn.NLLLoss() in one single class.)
* Backpropagation (i.e. the brunt of the work)

You can find my MLP code in `mlp.py`. As I continue to write more and more code from scratch, I expect my models performance to become worse and worse due to the advantages that using PyTorch has (it is written in C and highly optimized). For anyone who wants to implement a MLP in Python, and doesn't care for learning it indepthly, I highly recommend just completely using PyTorch. Maybe a future project is to do this from scratch in C!

`torch-mlp.py` is the same architecture, just built using all PyTorch functions.

## Architecture

My architecture is as follows:

* `Linear(784, 500)`
  * `ReLU()`
  * `Dropout()`
* `Linear(500, 250)`
  * `ReLU()`
  * `Dropout()`
* `Linear(250, 10)`
* `CrossEntropyLoss()`

## Results

I achieved a test accuracy of $0.9302$ with a loss of $0.2278$ on the [MNIST as JPG](https://www.kaggle.com/datasets/scolianni/mnistasjpg/data) dataset after $10$ epochs using gradient descent optimization. I used a learning rate of $0.001$ and a dropout probability of $0.1$. The weights for that model can be downloaded from `weights.pt`.

Using PyTorch functions and PyTorch's SGD optimizer in `torch-mlp.py`, I was able to achieve an accuracy of >$98\\perc$. Obviously, I am aim to match that with my own implementation.

## Future Goals

- [ ] Batching
- [ ] Implementing Softmax
- [ ] Implementing CrossEntropyLoss
- [ ] Implementing Backpropagation
- [ ] Writing my own matrix multiplication library
- [ ] Trying out cross-validation
- [ ] Trying out hyperparameter optimization

---
&copy; 2024 Colin Pannikkat
