"""
Elias Griffin
Project 4 - Neural Network

    This code currently doesn't work - when I run it on a dataset, it tends to
    give an accuracy of around 50%, but isn't constant - it ranges between 45% and
    60%. To me, this says that it might be an indexing issue, because it is
    clearly changing values, but it isn't changing them in a productive way.

Usage: python3 project3.py DATASET.csv

    self.weights is a matrix of weights that works as follows
        for a layer n, which denotes the weights between node layers
        n and n-1. if the preceding layer has 5 nodes, and the later layer has
        3 nodes, the matrix will be a 5x3 matrix
"""

import csv, sys, random, math, numpy

ALPHA = .1

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    #print(v1, v2)
    sum = 0

    for i in range(len(v1)):
        #print(v1[i], v2[i])
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def log_deriv(error):
    """returns the gradient(?) of some error - used in back propogating"""
    return logistic(error) * (1 - logistic(error))

def logistic_derivative(errors):
    """ above, but for use on lists"""
    return [log_deriv(error) for error in errors]

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class_binary()
        if class_prediction != y[0]:
            true_positives += 1

        #outputs = nn.get_outputs()
        #print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here

class NeuralNetwork():
    def __init__(self, dimens):

        #self.create_layers(layers)
        self.dimens = dimens
        self.weights, self.bias = self.create_params(dimens)
        self.input_matrix = {}


    def create_params(self, layer_sizes):
        """ Initializes matrix of weights and nodes
        self.nodes - a dictionary, where the layer maps to a list of nodes
        self.weights - a dictionary where each layer j maps to a size ixj matrix"""
        weights = {}
        bias = {}
        bias[0] = [1] * layer_sizes[0]

        for layer in range(1, len(layer_sizes)):
            weights[layer] = numpy.random.rand(layer_sizes[layer-1], layer_sizes[layer])
            bias[layer] = [1] * layer_sizes[layer]

        bias[len(layer_sizes)-1] = [0] * layer_sizes[-1]

        return weights, bias

    def predict_class_binary(self):
        """predicts a class of a binary output (a single node)"""
        return round(logistic(self.input_matrix[len(self.dimens)-1][0]))


    def activation_forward(self, act_matrix, weight_layer):
        """ Takes a list of activation inputs, and takes the dot product
        with the list of weights. Returns """
        biases = self.bias[weight_layer]

        act_matrix = dot_product(act_matrix, self.weights[weight_layer])

        return [act_matrix[i] + biases[i] for i in range(len(act_matrix))]

    def forward_propagate(self, input):
        """ takes an input, and propogates it through the network. Returns the output"""
        self.input_matrix[0] = input
        act_matrix = input
        for i in range(1, len(self.dimens)):

            #update our layer of activations
            act_matrix = self.activation_forward(act_matrix, i)

            #save our input values for use in back propogating
            self.input_matrix[i] = act_matrix

            #send the logistic function of all the inputs to the next layer
            act_matrix = [logistic(act) for act in act_matrix]

        return act_matrix

    def get_error(self, output,target):
        """ returns a list of errors given an output and an expected value"""
        return [target[i]-output[i] for i in range(len(output))]


    def node_layer_error(self, errors, layer):
        """ takes an error matrix and a layer, returns the individual errors
        of the node in the preceding layer based on the weights."""
        weights = self.weights[layer]

        node_errors = []

        for i, input in enumerate(self.input_matrix[layer-1]):
            node_errors.append(float(log_deriv(input) * dot_product(weights[i], errors)))

        return node_errors

    def update_weight(self, error_matrix, i, j):
        """ given a list of weights from a node to the next layer and errors
        from that next layer, updates the weights"""
        error = error_matrix[i-1][j]
        input = log_deriv(self.input_matrix[i-1][j])
        self.bias[i-1][j] += ALPHA * input * error
        self.weights[i][j] = [weight + ALPHA * input * error for weight in self.weights[i][j]]

    def back_propogate(self, output, target):
        """back-propogation algorithm """
        error_matrix = []

        #difference between output and expected
        output_error = self.get_error(output, target)

        #the opposite of every input in the output
        log_deriv = logistic_derivative(self.input_matrix[len(self.dimens)-1])


        #calculate errors for weights in output layer
        error_matrix = [[error * log_deriv[i] for i, error in enumerate(output_error)]]



        #calculate errors for rest of network
        for i in range(len(self.dimens)-1, 0, -1):
            #print(i)
            error_matrix.insert(0, self.node_layer_error(error_matrix[0], i))


        #update every weight, self.weights is categorized differently against
        #other stored matrices, hence the +1
        for i in range(1, len(self.weights)+1):

            #for every weight in layer of weights
            for j in range(len(self.weights[i])):
                self.update_weight(error_matrix, i, j)





    def back_propagation_learning(self, training):
        for example in training:
            input, target = example
            output = self.forward_propagate(input)
            self.back_propogate(output, target)

def test_3_bit(nn, training):
    """Test accuracy on a 3-bit incrementer"""
    accuracy = 0
    for example in training:
        input, target = example
        output =[round(num) for num in nn.forward_propagate(input)]
        #print(input, target, output)
        for i in range(len(output)):
            if output[i] == target[i]:
                accuracy += 1
    print('Accuracy: ', accuracy/24)

def hold_out_val(training):
    """ a simple hold-out validation"""
    random.shuffle(training)

    testing = training[:int(len(training)/2)]
    training = training[int(len(training)/2):]
    return testing, training


def test_three():
    """used to test the 3bit incrementer without switching the dataset taken in
    in command line"""
    header, data = read_data("increment-3-bit.csv", ",")
    nn = NeuralNetwork([3, 6, 3])
    training = convert_data_to_pairs(data, header)
    print(nn.forward_propagate([1,0,1]))
    for epoch in range(0):
        nn.back_propagation_learning(training)
    test_3_bit(nn, training)






def main():
    header, data = read_data(sys.argv[1], ",")

    paired_data = convert_data_to_pairs(data, header)



    nn = NeuralNetwork([2, 3, 1])

    f = open("data.csv", "a+")
    test_three()
    for epoch in range(10000):
        nn.input_matrix={}
        testing, training = hold_out_val(paired_data)
        nn.back_propagation_learning(training)
        acc = accuracy(nn, testing)
        #f.write("{},{}".format(epoch,acc))
        print("Accuracy on epoch {} is {} ".format(epoch, acc))

    f.close()


if __name__ == "__main__":
    main()
