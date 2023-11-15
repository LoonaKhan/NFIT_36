from neural_network import *


if __name__ == '__main__':
    # generate nn and dataset
    n = NeuralNetwork(input_size=728,
                      hidden_layers=2,
                      hidden_size=16,
                      output_size=10,
                      learning_rate=0.1
                      )
    #n.load()
    dataset = load_dataset()
    dataset[0] = [dataset[0][0]]
    dataset[1] = [dataset[1][0]]

    # try initial run
    out = n.feed_forward(dataset[0])

    start = time()
    for i in range(10000):
        n.train(dataset=dataset)
        # out = n.feed_forward(dataset[0])
        # n.back_propogation(dataset[0], dataset[1])
    print(f"trained in {time() - start} seconds\n")

    print("after training")
    out = n.feed_forward(dataset[0])
    for y, a in zip(out, dataset[1]):
        print(y)
        print(a)
        print("\n")
    n.save()
