from random import random
from sys import argv, maxint
import time
import numpy as np
import math
# http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/


class ImageFiles:
    def __init__(self):
        pass

    test_files = {}
    train_files = {}


def read_files(file_name):
    files = {}
    input_file = open(file_name, 'r')
    for line in input_file:
        data = line.split()
        img = np.empty((8, 8, 3), dtype=np.int)
        index = 2
        i = 0
        while i < 8:
            j = 0
            while j < 8:
                k = 0
                while k < 3:
                    img[i][j][k] = int(data[index])
                    index += 1
                    k += 1
                j += 1
            i += 1
        files[data[0] + data[1]] = {"orient": int(data[1]), "img": img}

    input_file.close()
    return files


def read_files2(file_name):
    files = {}
    input_file = open(file_name, 'r')
    for line in input_file:
        data = line.split()
        img = np.empty(192, dtype=np.int)
        index = 2
        i = 0
        while i < 192:
            img[i] = int(data[index])
            index += 1
            i += 1
        files[data[0] + data[1]] = {"orient": int(data[1]), "img": img}

    input_file.close()
    return files


def test_nearest():
    confusion_matrix = np.zeros((4, 4), dtype=np.int)
    i = 0
    result = 0
    for test_f_id in imf.test_files:
        i += 1
        test_f_img = imf.test_files[test_f_id]["img"]

        min_dist = maxint
        img_with_min_dist = ""

        for train_f_id in imf.train_files:
            train_f_img = imf.train_files[train_f_id]["img"]
            new_img = np.subtract(train_f_img, test_f_img)
            new_img = np.square(new_img)
            dist = np.sum(new_img)
            if dist < min_dist:
                min_dist = dist
                img_with_min_dist = train_f_id

        if imf.test_files[test_f_id]["orient"] == imf.train_files[img_with_min_dist]["orient"]:
            result += 1
        confusion_matrix[
            imf.test_files[test_f_id]["orient"] / 90, imf.train_files[img_with_min_dist]["orient"] / 90] += 1

    print "Confusion Matrix: \n" + str(confusion_matrix)
    print "Accuracy:" + str(result * 1.0 / (i * 1.0))


########################
# Neural Network
#########################################
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs) - 1):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def create_dataset():

    dataset = []
    for train_file_id in imf.train_files:
        my_list = [x for x in (imf.train_files[train_file_id]["img"])]
        dataset += [my_list + [imf.train_files[train_file_id]["orient"]/90]]
    return dataset


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

start_time = time.time()
train_file = argv[1]
test_file = argv[2]
mode = argv[3]

imf = ImageFiles()
imf.train_files = read_files2(train_file)
imf.test_files = read_files2(test_file)

if mode == "nearest":
    test_nearest()

if mode == "nnet":
    g_dataset = create_dataset()
    hidden_count = int(argv[4])
    print "Initializing"
    x_network = initialize_network(192, hidden_count, 4)
    print "Initialized"
    a = [1]*192
    label = [1]
    print "FP"
    output = forward_propagate(x_network, a + label)
    print "FP done"
    print "Train"
    train_network(x_network, g_dataset, 0.5, 20, 4)
    print "Trained"
    print "Check output"
    for test_file_id in imf.test_files:
        my_list = [x * 1.0 / 255.0 for x in (imf.test_files[test_file_id]["img"])]
        output = predict(x_network, my_list)
        print str(output) + " " + str(imf.test_files[test_file_id]["orient"]/90)

    print "Checked output"



end_time = time.time()
print end_time - start_time
