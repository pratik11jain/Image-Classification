from random import random, randint
from sys import argv, maxint
import numpy as np
import math
import copy
from cStringIO import StringIO


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
# For the given train and test data this method takes around 9-10 minutes
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
        nearest_file_str.write(
            train_f_id.split('.jpg')[0] + " " + str(imf.train_files[img_with_min_dist]["orient"]) + '\n')

    print "Confusion Matrix: \n" + str(confusion_matrix)
    print "Accuracy:" + str(result * 1.0 / (i * 1.0))
    with open('nearest_output.txt', 'w') as f:
        f.write(nearest_file_str.getvalue())


#################  Adaboost   ###############################
# This method takes around 40 seconds for 10 stumps and provides accuracy around 52%
# 9-10 minutes for 50 stumps and provides accuracy around 66%
# 35-40 minutes for 100 stumps and provides accuracy around 69%
# This function initializes the weight of each train example to 1/N where N is the size of train data
# This is the initialization used for Adaboost
def initializeWeight(train, totalCount):
    for example in train:
        train[example]["weight"] = 1.0 / totalCount


# For any instance of the train table this function gives the best suited attribute or attribute with min error
def getBestAttribute(boost, imageOrient):
    for trainFileId in imf.train_files:
        for pixels in boost:
            p = [int(pixel) for pixel in pixels.split()]
            if (imf.train_files[trainFileId]["img"][p[0]] > imf.train_files[trainFileId]["img"][p[1]] and
                        imf.train_files[trainFileId]["orient"] == imageOrient) or (
                    imf.train_files[trainFileId]["img"][p[0]] < imf.train_files[trainFileId]["img"][p[1]] and
                    imf.train_files[trainFileId]["orient"] != imageOrient):
                boost[pixels]["value"] += imf.train_files[trainFileId]["weight"]

    max_pixel = max([[pixel, boost[pixel]] for pixel in boost], key=lambda x: x[1]["value"])
    return max_pixel


# This function modifies the weights of the train examples that the selected attribute correctly identifies
def modifyWeight(modifier, stump_pixel, imageOrient):
    newSum = 0
    p = [int(pixel) for pixel in stump_pixel.split()]
    for trainFileId in imf.train_files:
        if (imf.train_files[trainFileId]["img"][p[0]] > imf.train_files[trainFileId]["img"][p[1]] and
                    imf.train_files[trainFileId]["orient"] == imageOrient) or (
                imf.train_files[trainFileId]["img"][p[0]] < imf.train_files[trainFileId]["img"][p[1]] and
                imf.train_files[trainFileId]["orient"] != imageOrient):
            imf.train_files[trainFileId]["weight"] *= modifier
        newSum += imf.train_files[trainFileId]["weight"]
    return newSum


# This function normalizes the weights in the train table
def normalize(normalizeValue):
    for trainFileId in imf.train_files:
        imf.train_files[trainFileId]["weight"] = imf.train_files[trainFileId]["weight"] / normalizeValue

# Create random weights network
def initialize_network(hidden_nodes):
    network = [[{'weights': [random() for i in range(193)]} for j in range(hidden_nodes)]]
    network.append([{'weights': [random() for i in range(hidden_nodes + 1)]} for j in range(4)])
    return network


def get_normalized_training_test_data(files):
    all_files = []

    for file_id in files:
        img = [x * 1.0 / 255.0 for x in files[file_id]["img"].tolist()]
        orient = files[file_id]["orient"] / 90

        all_files.append(img + [orient])
    return all_files


# Calculate neuron activation for an input
def neuron_output(weights, inputs):
    result = 0
    for i in range(len(weights) - 1):
        result += weights[i] * inputs[i]
    result += weights[i]
    return result

# Referred links
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

def neural_network():
    dataset = get_normalized_training_test_data(imf.train_files)
    test_dataset = get_normalized_training_test_data(imf.test_files)
    alpha = 0.4
    loops = 1
    hidden_nodes = int(argv[4])
    network = initialize_network(hidden_nodes)
    train_network(network, dataset, alpha, loops)
    results = test_data(test_dataset, network)

    print img_file_names
    with open ("nnet_output.txt","w") as f:
        for i in range(len(img_file_names)):
            f.write(str(img_file_names[i]).split(".txt")[0] + ".txt"+ " " + str(results[i]))


# Forward propagate input to a network output
def fwd_prop(network, img):
    for layer in network:
        result = []
        for neuron in layer:
            x = neuron_output(neuron['weights'], img)
            # Sigmoid function
            neuron['output'] = 1.0 / (1.0 + math.exp(-x))
            result.append(neuron['output'])
    return result


# Train network by running fwd_prop and back_prop on all images
def train_network(network, train, alpha, loops):
    for i in range(loops):
        for img in train:
            expected = [0] * 4
            expected[img[-1]] = 1
            fwd_prop(network, img)
            back_prop_error(network, expected)
            update_weights(network, img, alpha)


# Update weights at each layer
def update_weights(network, img, alpha):
    for neuron in network[0]:
        j = 0
        for j in range(192):
            neuron['weights'][j] += alpha * neuron['delta'] * img[j]
        neuron['weights'][j] += alpha * neuron['delta']

    inputs = [neuron['output'] for neuron in network[0]]

    for neuron in network[1]:
        j = 0
        for j in range(len(inputs) - 1):
            neuron['weights'][j] += alpha * neuron['delta'] * inputs[j]
        neuron['weights'][j] += alpha * neuron['delta']


# Back propagate error and store in neurons
def back_prop_error(network, expected):
    layer = network[1]
    errors = []
    for j in range(len(layer)):
        neuron = layer[j]
        errors.append(expected[j] - neuron['output'])

    for j in range(len(layer)):
        neuron = layer[j]
        neuron['delta'] = errors[j] * neuron['output'] * (1 - neuron['output'])
    layer = network[0]
    errors = []
    for j in range(len(layer)):
        error = 0.0
        for neuron in network[1]:
            error += (neuron['weights'][j] * neuron['delta'])
        errors.append(error)
    for j in range(len(layer)):
        neuron = layer[j]
        neuron['delta'] = errors[j] * neuron['output'] * (1 - neuron['output'])


def test_data(test, network):
    files_identified = 0
    confusion_matrix = np.zeros((4, 4), dtype=np.int)
    results = []
    for img in test:
        outputs = fwd_prop(network, img)
        result = outputs.index(max(outputs))
        results.append(result)
        confusion_matrix[img[-1]][result] += 1
        if result == img[-1]:
            files_identified += 1
    print "Accuracy::" + str(files_identified * 100.0 / len(test) * 1.0)
    print confusion_matrix
    return results

train_file = argv[1]
test_file = argv[2]
mode = argv[3]

imf = ImageFiles()
imf.train_files = read_files2(train_file)
imf.test_files = read_files2(test_file)

if mode == "nearest":
    # Uses the k nearest neighbor algortihm
    test_nearest()

# K- nearest code gives a slightly higher accuracy than Adaboost around 2-3% higher when decision stumps are 50
# But k-nearest is too overfitting and hence we use Adaboost as best option
# For best mode which is Adaboost of 50 stumps it takes 9-10 minutes to run the entire code
# The train part won't take more than 7-8 minutes and hence we have not used model-file
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
            pixel1 = randint(0, 191)
            pixel2 = randint(0, 191)
        imf.adaboost[str(pixel1) + " " + str(pixel2)] = {"value": 0}

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
            beta = (error) / (1 - error)

            # Stores the weights of the decision stump based on error
            # Some explanation for the formula used for weighting the decision stump
            # The formula is 1 + log(1-error/error)
            # The decision stump with least error rate or highest accuracy the one we select in the intial stages will have low denominator
            # This will lead to weight value above 1
            # The one with poor performance or most error will have negative values after taking log and thus such stumps will get weights between 0 and 1
            # The important thing is it will get below 1
            bestAttribute[stump].append(1 + math.log(1 / beta))
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
                p = [int(pixel) for pixel in decision_stump[0].split()]
                if (imf.test_files[testFileId]["img"][p[0]] > imf.test_files[testFileId]["img"][p[1]]):
                    decisionValue += decision_stump[2] * 1
                else:
                    decisionValue += decision_stump[2] * -1
            finalDecision[orient] = decisionValue
        decisionOrient = max([[key, finalDecision[key]] for key in finalDecision], key=lambda x: x[1])
        if imf.test_files[testFileId]["orient"] == decisionOrient[0]:
            count_correct += 1
        confusion_matrix[imf.test_files[testFileId]["orient"] / 90, decisionOrient[0] / 90] += 1
        file_str.write(testFileId.split('.jpg')[0] + ".jpg" + " " + str(decisionOrient[0]) + '\n')

    # Prints the confusion matrix, accuracy and writes the output in file in adaboost_output.txt
    print "Confusion Matrix: \n" + str(confusion_matrix)
    print "Accuracy:" + str((count_correct * 100.0) / len(imf.test_files))
    with open('adaboost_output.txt', 'w') as f:
        f.write(file_str.getvalue())


if mode == "nnet":
    img_file_names = []
    neural_network()
