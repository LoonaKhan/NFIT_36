from neural_network import *

if __name__ == '__main__':
    # creates nn and dataset
    n = NeuralNetwork(728,2,16,10,0.1)
    n.load()
    dataset = gen_dataset(100)

    for i in range(len(dataset)):
        out = n.feed_forward(dataset[i][0])
        p = f"{dataset[i][1]} [ "
        for j in range(len(out[0])):
            p += f"{round(out[0][j] - dataset[i][1][j], 4)}, "
        p += " ]"
        print(p)