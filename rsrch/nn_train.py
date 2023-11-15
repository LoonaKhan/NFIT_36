from neural_network import *


if __name__ == '__main__':
    # generate nn and dataset
    n = NeuralNetwork(input_size=728,
                      hidden_layers=2,
                      hidden_size=16,
                      output_size=10,
                      learning_rate=0.1
                      )
    n.load()
    dataset = load_dataset()

    # try initial run
    print("initial run")
    out = n.feed_forward(dataset[0][0])
    print(dataset[0][1])
    print(out,"\n")

    start = time()
    for i in range(1000):
        n.train(dataset=dataset)
    print(f"trained in {time() - start} seconds\n")

    print("after training")
    out = n.feed_forward(dataset[0][0])
    print(dataset[0][1])
    print(out)
    n.save()
