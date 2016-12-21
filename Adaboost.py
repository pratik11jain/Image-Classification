from random import random, randint
from sys import argv, maxint
import time
import numpy as np
import math
import copy
from cStringIO import StringIO
# http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

# To maintain all the train and test data files
class ImageFiles:
    def __init__(self):
        pass

    test_files = {}
    train_files = {}
    adaboost = {}
    orientation = [0, 90, 180, 270]

# This function reads the data from the file given as parameter and returns a numpy array of that data
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

# The k nearest neighbors algorithm is used to estimate the orientation of the test file
# For any one test data point euclidean distance is calculated with respect to every train example and the train example that gives min distance
# The orientation of that train example is labelled to the test data point
# This function prints accuracy, confusion matrix and writes the output in the file nearest_output.txt
def test_nearest():
    confusion_matrix = np.zeros((4, 4), dtype=np.int)
    i = 0
    result = 0
    nearest_file_str = StringIO()
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
        nearest_file_str.write(train_f_id.split('.jpg')[0] + " " + str(imf.train_files[img_with_min_dist]["orient"]) + '\n')

    print "Confusion Matrix: \n" + str(confusion_matrix)
    print "Accuracy:" + str(result * 1.0 / (i * 1.0))
    with open('nearest_output.txt', 'w') as f:
        f.write(nearest_file_str.getvalue())


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

# This function initializes the weight of each train example to 1/N where N is the size of train data
# This is the initialization used for Adaboost
def initializeWeight(train, totalCount):
	for example in train:
		train[example]["weight"] = 1.0/totalCount

# For any instance of the train table this function gives the best suited attribute or attribute with min error
def getBestAttribute(boost, imageOrient):
	for trainFileId in imf.train_files:
		for pixels in boost:
			p = [ int(pixel) for pixel in pixels.split()]
			if (imf.train_files[trainFileId]["img"][p[0]] > imf.train_files[trainFileId]["img"][p[1]] and imf.train_files[trainFileId]["orient"] == imageOrient) or (imf.train_files[trainFileId]["img"][p[0]] < imf.train_files[trainFileId]["img"][p[1]] and imf.train_files[trainFileId]["orient"] != imageOrient):
				boost[pixels]["value"] += imf.train_files[trainFileId]["weight"]

	max_pixel = max([[pixel, boost[pixel]] for pixel in boost], key= lambda x: x[1]["value"])
	return max_pixel

# This function modifies the weights of the train examples that the selected attribute correctly identifies
def modifyWeight(modifier, stump_pixel, imageOrient):
	newSum = 0
	p = [ int(pixel) for pixel in stump_pixel.split()]
	for trainFileId in imf.train_files:
		if (imf.train_files[trainFileId]["img"][p[0]] > imf.train_files[trainFileId]["img"][p[1]] and imf.train_files[trainFileId]["orient"] == imageOrient) or (imf.train_files[trainFileId]["img"][p[0]] < imf.train_files[trainFileId]["img"][p[1]] and imf.train_files[trainFileId]["orient"] != imageOrient):
			imf.train_files[trainFileId]["weight"] *= modifier
		newSum += imf.train_files[trainFileId]["weight"]
	return newSum

# This function normalizes the weights in the train table
def normalize(normalizeValue):
	for trainFileId in imf.train_files:
		imf.train_files[trainFileId]["weight"] = imf.train_files[trainFileId]["weight"]/normalizeValue

start_time = time.time()
train_file = argv[1]
test_file = argv[2]
mode = argv[3]

imf = ImageFiles()
imf.train_files = read_files2(train_file)
imf.test_files = read_files2(test_file)

if mode == "nearest":
	#Uses the k nearest neighbor algortihm
    test_nearest()

# K- nearest code gives a slightly higher accuracy than Adaboost around 2-3% higher when decision stumps are 50
# But k-nearest is too overfitting and hence we use Adaboost as best option
if mode == "adaboost" or mode == "best":
	# If mode is best we use Adaboost with 50 decision stumps
    stump_count = 50 if mode == "best" else int(argv[4])
    count_Train = len(imf.train_files)

    # Creates the dictionary of pair of pixels
    # Length of dictionary is equal to stump size
    # The pair of pixels are selected randomly 
    for i in range(0, stump_count):
        pixel1 = -1
        pixel2 = -1
        while pixel1 == -1 or pixel2 == -1 or (str(pixel1) + " " + str(pixel2)) in imf.adaboost:
            pixel1 = randint(0,191)
            pixel2 = randint(0,191)
        imf.adaboost[str(pixel1) + " " + str(pixel2)] = {"value": 0};

    # Adaboost method is used to create an ensembler for each orientation individually
    all_orientation_stump = {}
    for orient in imf.orientation:
    	bestAttribute = []
    	initializeWeight(imf.train_files, count_Train)

    	newBoost = copy.deepcopy(imf.adaboost)
    	for stump in range(0, stump_count):
    		bestAttribute.append(getBestAttribute(newBoost, orient))
    		totalWeight = sum([imf.train_files[train]["weight"] for train in imf.train_files])
    		error = (totalWeight - bestAttribute[stump][1]["value"]) / totalWeight
    		# Sometimes the pair of pixel selected randomly are so poor that the error rate is 100% so we give error rate 99% to avoid divide by zero error
    		error = 0.99 if error == 1 else error

    		# Calculates the modified weights to be assigned
    		beta = (error)/(1-error)

    		# Stores the weights of the decision stump based on error
    		bestAttribute[stump].append(1 + math.log(1/beta))
    		normalizeSum = modifyWeight(beta, bestAttribute[stump][0], orient)
    		normalize(normalizeSum)
    		del newBoost[bestAttribute[stump][0]]

    		for key in newBoost:
    			newBoost[key]["value"] = 0

    	all_orientation_stump[orient] = bestAttribute

    # Essembler for each orientation is executed on any test data point
    # The one that gives best value is used to label the test point 
    count_correct = 0
    confusion_matrix = np.zeros((4, 4), dtype=np.int)
    file_str = StringIO()
    for testFileId in imf.test_files:
    	finalDecision = {}
    	for orient in all_orientation_stump:
    		decisionValue = 0
    		for decision_stump in all_orientation_stump[orient]:
    			p = [ int(pixel) for pixel in decision_stump[0].split()]
    			if (imf.test_files[testFileId]["img"][p[0]] > imf.test_files[testFileId]["img"][p[1]]):
    				decisionValue += decision_stump[2] * 1
    			else:
    				decisionValue += decision_stump[2] * -1
    		finalDecision[orient] = decisionValue
    	decisionOrient = max([ [key, finalDecision[key]] for key in finalDecision], key = lambda x: x[1])
    	if imf.test_files[testFileId]["orient"] == decisionOrient[0]:
    		count_correct += 1
    	confusion_matrix[imf.test_files[testFileId]["orient"] / 90, decisionOrient[0] / 90] += 1
    	file_str.write(testFileId.split('.jpg')[0] + ".jpg" + " " + str(decisionOrient[0]) + '\n')

    # Prints the confusion matrix, accuracy and writes the output in file in adaboost_output.txt
    print "Confusion Matrix: \n" + str(confusion_matrix)
    print "Accuracy:" + str( (count_correct * 100.0) / len(imf.test_files) )
    with open('adaboost_output.txt', 'w') as f:
        f.write(file_str.getvalue())



    
        

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
