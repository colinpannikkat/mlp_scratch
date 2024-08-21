from mlp import MLP
from dataset import MNISTDataset, TitanicDataset
from tqdm import tqdm

def mnist():
    mlp = MLP([
                (784, 500),
                (500, 250),
                (250, 10)
            ],
            dp=0.2
            )
    data = MNISTDataset(data_pth="MNIST-jpg_dataset")

    train_data, test_data = data.split_shuffle(0.8)
    mlp.train(train_data, epochs=10, lr=0.001)

    mlp.eval()
    test_accuracy = []
    tot_test_loss = 0
    
    print("-----------------Test-----------------")
    for _, idx in tqdm(enumerate(range(len(test_data))), total=len(test_data)):
        loss, good = mlp.forward(test_data[idx])

        tot_test_loss += loss.item()
        test_accuracy.append(good)

    print(f"Test Set Loss: ", tot_test_loss/len(test_data))
    print(f"Test Accuracy: {sum(test_accuracy)/len(test_accuracy)}")
    print("--------------------------------------")

    # mlp.save_weights("weights.pt")

def titanic():

    mlp = MLP([
                (7, 100),
                (100, 2)
            ],
            dp=0.0
            )
    data = TitanicDataset(data_pth="titanic_dataset")

    train_data, test_data = data.split_shuffle(0.8)
    mlp.train(train_data, epochs=50, lr=0.0001, train_val_split=0.9)

    mlp.eval()
    test_accuracy = []
    tot_test_loss = 0
    
    print("-----------------Test-----------------")
    for _, idx in tqdm(enumerate(range(len(test_data))), total=len(test_data)):
        loss, good = mlp.forward(test_data[idx])

        tot_test_loss += loss.item()
        test_accuracy.append(good)

    print(f"Test Set Loss: ", tot_test_loss/len(test_data))
    print(f"Test Accuracy: {sum(test_accuracy)/len(test_accuracy)}")
    print("--------------------------------------")

    # mlp.save_weights("weights.pt")

if __name__ == "__main__":
    titanic()