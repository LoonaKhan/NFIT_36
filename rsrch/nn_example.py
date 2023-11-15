"""
This was a GPT generated example of a back propogation implementation as the code implementation is quite complex compared to the math i did on paper.
I used it as a reference only. you can tell because this example uses more conventional matrices instead of 2d arrays.
I am keeping it here for now for future reference.
"""

import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
    
    def feedforward(self, inputs):
        # Calculate hidden layer output
        self.hidden = sigmoid(np.dot(inputs, self.weights_input_hidden))
        # Calculate output
        self.output = sigmoid(np.dot(self.hidden, self.weights_hidden_output))
        return self.output
    
    def backpropagation(self, inputs, target, learning_rate):
        # Calculate output error
        output_error = target - self.output
        # Calculate output delta
        output_delta = output_error * sigmoid_derivative(self.output)
        print(output_delta)
        
        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        # Calculate hidden layer delta
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)
        
        # Update weights for hidden to output
        self.weights_hidden_output += self.hidden.T.dot(output_delta) * learning_rate
        # Update weights for input to hidden
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate

# Example usage
# Define input, hidden, and output layer sizes
input_size = 3
hidden_size = 4
output_size = 1

# Create a neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Example training data
training_inputs = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [1, 1, 0],
                            [0, 1, 1]])
training_outputs = np.array([[0, 1, 1, 0]]).T

# Train the neural network
for i in range(1):
    nn.feedforward(training_inputs)
    nn.backpropagation(training_inputs, training_outputs, 0.1)

# Test the trained network
test_input = np.array([1, 1, 1])
predicted_output = nn.feedforward(test_input)
print("Predicted output:", predicted_output)

