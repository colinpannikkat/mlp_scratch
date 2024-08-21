from mlp import MLP, Dataset
from tqdm import tqdm

def main():
    mlp = MLP(dp=0.2)
    data = Dataset(data_pth="MNIST-jpg_dataset")

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

if __name__ == "__main__":
    main()