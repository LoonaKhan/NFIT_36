from neural_network import *

if __name__ == '__main__':
    # creates nn and dataset
    n = NeuralNetwork(728,2,100,10,0.1)
    n.load()
    dataset = load_dataset()
    dataset[0] = [dataset[0][0]]
    dataset[1] = [dataset[1][0]]

    out = n.feed_forward(dataset[0][0])
    a = dataset[1][0]
    print("expected: ", a)
    print("output: ", out)
