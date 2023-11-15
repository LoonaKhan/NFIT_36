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

    def process(self, inputs: list[float]):
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

            out = self.process(sample)
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




if __name__ == '__main__':

    # learning rate

    # creates the network 
    net = Network(3,3,3,1)
    # print(net)

    with open("test_dataset.json", "r") as f:
        dataset = json.load(f)

    start = time()
    net.learn_dataset(dataset, True)
    num_runs = 100_000
    for i in range(num_runs):
        net.learn_dataset(dataset)
        #print(f"{round(i/num_runs*100, 2)}% complete training")

    net.learn_dataset(dataset, True)

    print(f"Completed in {time() - start} seconds")
    #
    # target = [0,1,0]
    #
    # # processes our inputs
    # inputs = [3,3,3]
    # print("inputs: ", inputs)
    # print("target: ", target)
    #
    # out = net.process(inputs)
    # pstring = "[ "
    # for n in out:
    #     pstring += str(n.output) + ", "
    # pstring += " ]"
    # print("\ninitial run")
    # print(pstring + "\n")
    #
    # for i in range(3):
    #
    #     # prints our inputs and outputs
    #     pstring = "[ "
    #     for n in out:
    #         pstring += str(n.output) + ", "
    #     pstring += " ]"
    #     net.back_prop(target)
    #     #print(net)
    #     out = net.process(inputs)
    #
    # print(f"after {i} training sessions:")
    # print(pstring + "\n")
