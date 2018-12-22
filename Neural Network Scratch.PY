# used for working with matrices
import numpy as np

# used to import sigmoid function
import scipy.special 

# used for plotting
import matplotlib.pyplot as plt
import time

class Neural_Network_1H():

    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr):

        # Values for neural network
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.lr = lr

        # Makes matrix of shape of values above
        # assigned random weights

        # Don't think about the nodes, thing about the weights between the layers

        # Each Row is a Node and Each Column is an input to the node

        # This is densenly connected, so each hidden node has n inputs (same as columns) 
        self.input_to_hidden = (np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)
        self.hidden_to_output = (np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)

        # Ditermines what will be sent over weight
        self.activation = lambda x: scipy.special.expit(x)
        pass

    def train(self, input_vector, target_vector):

        # Makes vector into 784 x 1 matrix, and transposes to feed into network
        inputs = np.array(input_vector, ndmin=2).T
        targets = np.array(target_vector, ndmin=2).T

        # Feed inputs into Hidden Layer
        # Applying activation to each output
        feed_hidden = np.dot(self.input_to_hidden, inputs)
        feed_hidden_output = self.activation(feed_hidden)
   
        
        # Feeds output from hidden into last layer
        # Applies activation to each row of output to normlizer from 0.01 - 0.99
        network_output = np.dot(self.hidden_to_output, feed_hidden_output)
        final_outputs = self.activation(network_output)
    
        
        # Sees difference between outputs and targets
        error = targets - final_outputs
        # w*e/sum or weights sending porpotional error back to hidden
        # Assigning porportional error to each node/row (weight/sum_weights) * total_error = porpotional error
        hidden_errors = np.dot(self.hidden_to_output.T, error)

        # NOW EACH HIDDEN_TO_OUTPUT ROW/NODE HAS AN ERROR ASSIGNED TO IT!
        # HIDDEN_ERRORS IS USED AS A REFRENCE 
        
        # PARTIAL DERIVATIVE FOR ERROR SLOPE

        # update the weights for the links between the input and hidden layers
        self.input_to_hidden += self.lr * np.dot((hidden_errors * feed_hidden_output * (1.0 - feed_hidden_output)), np.transpose(inputs))
        
        # update the weights for the links between the hidden and output layers
        self.hidden_to_output += self.lr * np.dot((error * final_outputs * (1.0 - final_outputs)), np.transpose(feed_hidden_output))

        pass
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.input_to_hidden, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.hidden_to_output, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation(final_inputs)
        
        return final_outputs



input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1

# create net
n = Neural_Network_1H(input_nodes, hidden_nodes, output_nodes, learning_rate)

# loading data
data_file = open("mnist_train.csv", 'r')
data_list = data_file.readlines()
data_file.close()

# this section goes through each entry and trains data

#######################################################
# THIS PART OF THE CODE CHANGES THE NEURAL NETWORK!!!!!!!!
#######################################################

print("Started Training for 5 Epochs\n")
whole_start = time.time()
# Running 5 epochs on data
for i in range(1, 6):
    epoch_start = time.time()
    print("Epoch {}".format(i))
    for image in data_list:
        # converting in normal list
        all_values = image.split(",")
    
        # Scaling all values between 0.01 and 1.00 color of pixel
        # In index 0 is label
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # creates an 1D array of 10 zeros, and adds 0.01 to each entry
        targets = np.zeros(output_nodes) + 0.01

        # Label stored in index 0 of image
        label = int(all_values[0])

        # Index and label matches up, so make that index position equal to 0.99
        targets[label] = 0.99
    
        # sending 2 vectors to network
        n.train(inputs, targets)

    epoch_end = time.time()
    print("Time: {}\n".format((epoch_end-epoch_start)))
    

print("Training Ended\n")

# loading test files
test_file = open("mnist_test.csv", 'r')
test_list = test_file.readlines()
test_file.close()

correct = []
wrong = []
total_test = 0

print("Testing Started\n")
for record in test_list:
    # converting in normal list
    all_values = record.split(",")
    
    # correct label from record
    correct_lable = int(all_values[0])
    
    # Scaling all values between 0.01 and 1.00 color of pixel
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    # query network
    output = n.query(inputs)
    
    total_test = total_test + 1
    
    # get the index of the max number from returned numpy array
    lable = np.argmax(output)
    
    # if it matches, append correct list
    if lable == correct_lable:
        correct.append(1)
    else:
        wrong.append(0)
        
whole_end = time.time()

print("Total Time: {}\n".format(whole_end-whole_start))
        
print("Performance: {}".format(len(correct)/total_test))

print("Out of {} tests, {} predictions were correct".format(total_test, len(correct)))
