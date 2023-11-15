"""
Base code for Neural Network implementation.

Python Object Oriented Programming (POOP) is used here. its easy to follow if you arent familiar, but its just a bit easier if you are.

Comments will explain how the network works and the implementation.
"""
import json
from random import uniform, randint # uniform generates floats and randint generates integers
import math
import numpy as np
from time import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Node:
    def __init__(self, last_layer_size : int=0): # node constructor
        """
            all nodes have outputs, thresholds and a list of weights.

            threshold is initially set to a random number
            Each weight points to a node in the previous layer.

            we pass in last_layer_size so we know how many weights to assign
            Ex: 3 nodes in the previous layer, so we create 3 weights.
            If this is the first layer, we have no weights.
        """
        self.output = 0.0
        self.threshold = uniform(0.0, 1.0)
        self.weights = []
        self.weight_nudges = []

        for n in range(last_layer_size): # fill up weights 
            self.weights.append(uniform(-1.0, 1.0))
            self.weight_nudges.append([])

    def activate(self, last_layer=None, input :float=None):
        """
        activation method. I just used the sigmoid function here.

        We take the previous layer's outputs, last_layer and multiply each output by the corresponding weight and then subtract the threshold and pass it into the sigmoid function, 1/(1+e^-x)

        However, if this node is in the first layer, we directly output the input.
        """

        if input is not None: # if the node is given an input.
            self.output = input
            return

        # otherwise
        z = 0
        for n,w in zip(last_layer, self.weights): # for each node in the last layer and weight in weights 
            z += n.output * w
        z -= self.threshold
        self.output = sigmoid(z) # sigmoid function to activate

    def __str__(self): # debug fuction that prints important info on the node 
        return f"(t: {self.threshold}, w: {self.weights}) "

    def adjust_weights(self):
        for w in range(len(self.weights)):
            avg = 0
            for i in range(len(self.weight_nudges[w])):
                avg += self.weight_nudges[w][i]
            avg /= i
            self.weights[w] += avg

    def clear_weight_nudges(self): # clears weight nudges. meant for when we retrain over a dataset
        for w in self.weights:
            self.weight_nudges.append([])

class Network:
    def __init__(self,inputs, outputs, layer_size, hidden_layers):
        """
        networks have a list of nodes. this is a 2d array as nodes are organized in layers.

        when we create a network, we specify the size of the input and output layers seperately.
        layer_size is the size for default/hidden layers.
        lastly we specify how many hidden layers we want

        Ex: if i specified a network with inputs=3, outputs=2, layer_size=4, hidden_layers=2
        it would look like: https://imgur.com/a/dvGaomv
        """

        # input layer 
        self.nodes = [[]]
        for i in range(inputs):
            self.nodes[0].append(Node())

        # hidden layers 
        for i in range(1, hidden_layers+1):
            self.nodes.append([])
            for j in range(layer_size):
                self.nodes[i].append(Node(len(self.nodes[i-1])))

        # output layer 
        self.nodes.append([])
        for i in range(outputs):
            self.nodes[hidden_layers+1].append(Node(len(self.nodes[hidden_layers])))

        self.learning_rate = 0.1

    def feed_forward(self, inputs: list[float]):
        """
        gives the network a set of inputs, which the network computes and returns an output, the outputs of the last layer.

        The first layer takes i the inputs given, while other layers are fed the previous layer.
        """
        for l in range(len(self.nodes)):
            for n in range(len(self.nodes[l])):
                if l == 0:
                    self.nodes[l][n].activate(input=inputs[n])
                else:
                    self.nodes[l][n].activate(last_layer=self.nodes[l-1])

        return self.nodes[len(self.nodes)-1]

    def __str__(self): # debug function to print the important info of the network 
        ret = ""

        for l in range(len(self.nodes)):
            ret += f"L{l}: [ "
            for n in range(len(self.nodes[l])):
                ret += str(n)+self.nodes[l][n].__str__()
            ret += "]\n"

        return ret

    def back_prop(self, target):
        """
        for all training data in a training set, determine the nudges to be made to all weights
        todo:
            rewrite this using matrices?
            more efficient, scalable and maybe better results?

        determine the nudge for a weight of a node, for all nodes
            find the error of each node:
                output layer nodes: target - output
                other layer nodes: delta of the last node * output

            find the delta of each node:
                error of the node * sigmoid_derivative of the node

            adjust the weights of each node:
                w  = w + previous node's output * delta of the node * learning rate

            start at the end of the network and use back propogation to continue backward and analyze all weights in the network.
        """
        deltas = []
        for i in range(len(self.nodes)): # fill up delta with blank arrays
            deltas.append([])

        for l in range(len(self.nodes)-1, 0, -1): # cycles through each layer.
            errors = []

            # finds the errors
            if l == 0: # input layer doesnt have weights
                continue
            elif l == len(self.nodes)-1:
                for n in range(len(self.nodes[l])):
                    errors.append(target[n] - self.nodes[l][n].output)
            else:
                for n in range(len(self.nodes[l])):
                    errors.append(deltas[l+1][n] * self.nodes[l+1][n].output) # error? if l+1 is smaller TODO: fix for larger hidden layers than output layer

            for n in range(len(self.nodes[l])): # deltas
                deltas[l].append(errors[n] * sigmoid_derivative(self.nodes[l][n].output))

            for n in range(len(self.nodes[l])): # weight adjustments. todo: add to weight_nudges here
                for w in range(len(self.nodes[l][n].weights)):
                    self.nodes[l][n].weight_nudges[w].append( self.nodes[l-1][n].output * deltas[l][n] * self.learning_rate)

    def learn_dataset(self, dataset, debug=False):
        """
        Still need a dataset and its format to implement this

        Iterates over a dataset and feeds forward and back propogates for each data sample.
        from each data sample, it stores the weights nudges in a seperate matrix.
        after going over the entire dataset, we change the NN's weights by averaging the weights of each data sample nudge.
        """

        # clears out the weight nudges of each node in the network
        for l in range(len(self.nodes)):
            for n in range(len(self.nodes[l])):
                self.nodes[l][n].clear_weight_nudges()

        # feed forward and back propogate for each sample in the dataset
        for sample, answer in dataset:
            if debug is True: print(sample)
            if debug: print(answer)

            out = self.feed_forward(sample)
            pstring = "[ "
            for n in out:
                pstring += str(n.output) + ", "
            pstring += " ]"
            if debug: print(pstring+"\n")

            self.back_prop(answer)

        # adjust the node weights
        for l in range(len(self.nodes)):
            for n in range(len(self.nodes[l])):
                self.nodes[l][n].adjust_weights()


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, learning_rate):
        self.learning_rate = learning_rate

        self.weights = []
        self.weights.append(np.random.rand(input_size, hidden_size)*2 - 1)
        for i in range(hidden_layers):
            self.weights.append(np.random.rand(hidden_size, hidden_size)*2 - 1)
        self.weights.append(np.random.rand(hidden_size, output_size)*2 - 1)
        #self.weights = np.array(self.weights)

        self.weight_adjustments = []
        for i in range(len(self.weights)):
            self.weight_adjustments.append(np.zeros([len(self.weights[i]),len(self.weights[i][0])]))


    def feed_forward(self, inputs):
        self.outputs=[]
        for i in range(len(self.weights)):
            if i ==0:
                output = sigmoid(np.dot(inputs, self.weights[0]))
            else:
                output = sigmoid(np.dot(self.outputs[i-1], self.weights[i]))

            self.outputs.append(output)

        return output

    def back_propogation(self, inputs, target):
        self.deltas = []
        for i in range(len(self.outputs)):
            self.deltas.append([])

        for i in range(len(self.outputs)-1, -1, -1):
            if i == len(self.outputs)-1:
                err = target - self.outputs[i]
            else:
                err = np.dot(self.deltas[i+1], self.weights[i+1].T)
            self.deltas[i] = (err * sigmoid_derivative(self.outputs[i]))

        for l in range(len(self.weights)-1, -1, -1):
            if l == 0:
                self.weight_adjustments[l] += inputs.T.dot(self.deltas[l]) * self.learning_rate
            else:
                self.weight_adjustments[l] += self.outputs[l-1].T.dot(self.deltas[l]) * self.learning_rate

    def train(self, dataset):
        # clear weight adjustments.
        self.weight_adjustments = []
        for i in range(len(self.weights)):
            self.weight_adjustments.append(np.zeros([len(self.weights[i]), len(self.weights[i][0])]))

        for sample, ans in zip(dataset[0], dataset[1]):
            self.feed_forward(sample)
            self.back_propogation(sample, ans)

        for w in range(len(self.weights)):
            self.weights[w] += self.weight_adjustments[w] / len(dataset)

    def save(self):
        weights_copy = self.weights
        for l in range(len(weights_copy)):
            weights_copy[l] = weights_copy[l].tolist()
        with open("saved_nn.json", "w") as f:
            json.dump(weights_copy, f)

    def load(self):
        with open("saved_nn.json", "r") as f:
            self.weights = json.load(f)
        for l in range(len(self.weights)):
            self.weights[l] = np.array(self.weights[l])


def gen_dataset(num_samples, inputs, outputs):
    dataset = [[],[]]
    for i in range(num_samples):
        dataset[0].append(np.random.random([inputs,1]).T)

        ans = []
        added_one = False
        for j in range(outputs):
            if not added_one and (randint(0,outputs) == 0 or j ==outputs-1):
                ans.append(1.0)
                added_one = True
            else:
                ans.append(0.0)
        dataset[1].append(ans)

    return dataset

def save_dataset(dataset):
    data = [[],[]]
    for s in range(len(dataset[0])):
        data[0].append(dataset[0][s].tolist())
        data[1].append(dataset[1][s])
    with open("test_dataset.json", "w") as f:
        json.dump(data, f)

def load_dataset():
    with open("test_dataset.json", "r") as f:
        data =  json.load(f)

    for s in range(len(data[0])):
        data[0][s] = np.array(data[0][s])

    return data


if __name__ == '__main__':



    n = NeuralNetwork(728,2,16,10, 0.1)
    nn = Network(728, 10, 10,2)

    #n.load()
    dataset = gen_dataset(100)

    # start = time()
    # for i in range(100):
    #     n.train(dataset=dataset)
    # print(f"trained in {time() - start} seconds")

    for i in range(len(dataset)):
        out = n.feed_forward(dataset[i][0])
        p = f"{dataset[i][1]} [ "
        for j in range(len(out[0])):
            p += f"{round(out[0][j] - dataset[i][1][j], 4)}, "
        p += " ]"
        print(p)
