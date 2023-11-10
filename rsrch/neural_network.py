"""
Base code for Neural Network implementation.

Python Object Oriented Programming (POOP) is used here. its easy to follow if you arent familiar, but its just a bit easier if you are.

Comments will explain how the network works and the implementation.
"""

from random import uniform, randint # uniform generates floats and randint generates integers
import math


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
        self.z = 0.0
        self.delta=0
        self.weights = []

        for n in range(last_layer_size): # fill up weights 
            self.weights.append(uniform(-1.0, 1.0))

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
        for n,w in zip(last_layer, self.weights): # for each node in the last layer and weight in weights 
            self.z += n.output * w
        self.z -= self.threshold
        self.output = 1/(1 + math.exp(-self.z)) # sigmoid function to activate

    def __str__(self): # debug fuction that prints important info on the node 
        return f"(t: {self.threshold}, w: {self.weights}) "

    def sigmoid_derivative(self, z): # we use this for calculating back propogation
        return 1 / (1 + math.exp(-z))

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

        self.learning_rate = 0.1 # how fast our NN learns

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

    def find_deltas(self, y : list[float]): # back prop find all weights
        for l in range(len(self.nodes), 0):
            if l == len(self.nodes)-1: # last lyr
                for n in range(len(self.nodes[l]), 0):
                    self.nodes[l][n].delta = (self.nodes[l][n].output - y[l]) * (self.nodes[l][n].output * (1 - self.nodes[l][n].output)) # last part is transfer derivative, was y - output before
            else: # we get all deltas of the next layer * each weight
                # add in delta calculations
                for n in range(len(self.nodes[l])):
                    err = 0.0
                    for next_lyr_node in ((self.nodes[l+1])):
                        err += next_lyr_node.weights[n] * next_lyr_node.delta
                    self.nodes[l][n].delta = err * (self.nodes[l][n].output * (1 - self.nodes[l][n].output))


    def adjust_weights(self):
        for l in range(1, len(self.nodes)):
            for n in range(len(self.nodes[l])):
                for w in range(len(self.nodes[l][n].weights)):
                    self.nodes[l][n].weights[w] -= self.nodes[l][n].delta * self.nodes[l][n].z * self.learning_rate # self.nodes[l-1][w].output


    def back_prop(self):
        """
        WIP

        for all training data in a training set, determine the nudges to be made to all weights
        3d array?            each weight for each node in each layer

        determine the nudge for a node
            the output of the node paired with it
            the derivative of sigmoid of all the resulting z's
            2*(actual - expected)

            start at the end of the network and use back propogation to continue backward and analyze all weights in the network.
        """
        pass



if __name__ == '__main__':

    # creates the network 
    net = Network(3,3,3,1)
    print(net)

    # processes our inputs 
    inputs = [3,3,3]
    y = [0,1,0]
    out = net.process(inputs)


    # prints our inputs and outputs 
    print("inputs: ", inputs)
    pstring = "[ "
    for n in out:
        pstring+= str(n.output)+", "
    pstring +=" ]"
    print(pstring)
    print("expected", y)

    # back prop

    for i in range(10):
        net.find_deltas(y)
        net.adjust_weights()
        out = net.process(inputs)

        pstring = "[ "
        for n in out:
            pstring += str(n.output) + ", "
        pstring += "]"
        print(pstring)

