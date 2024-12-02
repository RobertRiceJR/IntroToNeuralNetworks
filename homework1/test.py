from csv import reader                 # reader object reads a csv file line by line
from random import seed                # seeds the random number generator
from random import random, shuffle     # random functions for weights and shuffling
# Removed the import statement since we define the Perceptron class in this script
# from Perceptron import Perceptron      # this is the Perceptron class in the Perceptron.py file

######################################################################
##### DATASET FUNCTIONS                                          #####
######################################################################

# Load the CSV file containing the inputs and desired outputs
def load_csv(filename):
    # dataset will be the matrix containing the inputs
    dataset = list()

    # Standard Python code to read each line of text from the file as a row
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue

            # add current row to dataset
            dataset.append(row)

    return dataset

# Convert the input values in the specified column of the dataset from strings to floats
def convert_inputs_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert the desired output values, located in the specified column, to unique integers
# For 2 classes of outputs, 1 desired output will be 0, the other will be 1
def convert_desired_outputs_to_int(dataset, column):
    # Enumerate all the values in the specified column for each row
    class_values = [row[column] for row in dataset]

    # Create a set containing only the unique values
    unique = set(class_values)

    # Create a lookup table to map each unique value to an integer (either 0 or 1)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i

    # Replace the desired output string values with the corresponding integer values
    for row in dataset:
        row[column] = lookup[row[column]]
    
    return lookup

# Create the training and test sets
def create_training_set(dataset, split_ratio=0.7):
    shuffle(dataset)
    split = int(len(dataset) * split_ratio)
    training_set = dataset[:split]
    test_set = dataset[split:]
    return training_set, test_set

######################################################################
##### PERCEPTRON CLASS IMPLEMENTATION                            #####
######################################################################

class Perceptron(object):

    # Create a new Perceptron
    # 
    # Params:   bias -  arbitrarily chosen value that affects the overall output
    #                   regardless of the inputs
    #
    #           synaptic_weights -   list of initial synaptic weights for this Perceptron
    def __init__(self, bias, synaptic_weights):
        self.bias = bias
        self.synaptic_weights = synaptic_weights

    # Activation function
    #   Quantizes the induced local field
    #
    # Params:   z - the value of the induced local field
    #
    # Returns:  an integer that corresponds to one of the two possible output values (usually 0 or 1)
    def activation_function(self, z):
        return 1 if z >= 0 else 0

    # Compute and return the weighted sum of all inputs (not including bias)
    #
    # Params:   inputs - a single input vector (which may contain multiple individual inputs)
    #
    # Returns:  a float value equal to the sum of each input multiplied by its
    #           corresponding synaptic weight
    def weighted_sum_inputs(self, inputs):
        return sum(w * x for w, x in zip(self.synaptic_weights, inputs))

    # Compute the induced local field (the weighted sum of the inputs + the bias)
    #
    # Params:   inputs - a single input vector (which may contain multiple individual inputs)
    #
    # Returns:  the sum of the weighted inputs adjusted by the bias
    def induced_local_field(self, inputs):
        return self.weighted_sum_inputs(inputs) + self.bias

    # Predict the output for the specified input vector
    #
    # Params:   input_vector - a vector or row containing a collection of individual inputs
    #
    # Returns:  an integer value representing the final output, which must be one of the two
    #           possible output values (usually 0 or 1)
    def predict(self, input_vector):
        z = self.induced_local_field(input_vector)
        return self.activation_function(z)

    # Train this Perceptron
    #
    # Params:   training_set - a collection of input vectors that represents a subset of the entire dataset
    #           learning_rate_parameter -    the amount by which to adjust the synaptic weights following an
    #                                        incorrect prediction
    #           number_of_epochs -  the number of times the entire training set is processed by the perceptron
    #
    # Returns:  no return value
    def train(self, training_set, learning_rate_parameter, number_of_epochs):
        for epoch in range(number_of_epochs):
            for row in training_set:
                inputs = row[:-1]
                desired_output = row[-1]
                prediction = self.predict(inputs)
                error = desired_output - prediction
                # Update weights
                for i in range(len(self.synaptic_weights)):
                    self.synaptic_weights[i] += learning_rate_parameter * error * inputs[i]
                # Update bias
                self.bias += learning_rate_parameter * error

    # Test this Perceptron
    # Params:   test_set - the set of input vectors to be used to test the perceptron after it has been trained
    #
    # Returns:  a collection or list containing the actual output (i.e., prediction) for each input vector
    def test(self, test_set):
        predictions = []
        for row in test_set:
            inputs = row[:-1]
            prediction = self.predict(inputs)
            predictions.append(prediction)
        return predictions

######################################################################
##### CREATE A PERCEPTRON, TRAIN IT, AND TEST IT                 #####
######################################################################

# Seed random number generator for reproducibility
seed(1)

# Step 1: Acquire the dataset
dataset = load_csv('sonar_all-data.csv')

# Step 2: Convert the string input values to floats
for column in range(len(dataset[0])-1):
    convert_inputs_to_float(dataset, column)

# Step 3: Convert the desired outputs to int values
lookup = convert_desired_outputs_to_int(dataset, len(dataset[0])-1)

# Step 4: Create the training set
training_set, test_set = create_training_set(dataset)

# Step 5: Create the perceptron
num_inputs = len(dataset[0]) - 1
synaptic_weights = [random() - 0.5 for _ in range(num_inputs)]
bias = random() - 0.5
perceptron = Perceptron(bias, synaptic_weights)

# Step 6: Train the perceptron
learning_rate_parameter = 0.1
number_of_epochs = 100
perceptron.train(training_set, learning_rate_parameter, number_of_epochs)

# Step 7: Test the trained perceptron
predictions = perceptron.test(test_set)

# Step 8: Display the test results and accuracy of the perceptron
actual_outputs = [row[-1] for row in test_set]
correct = sum(1 for actual, predicted in zip(actual_outputs, predictions) if actual == predicted)
accuracy = correct / len(actual_outputs) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Optionally, display individual predictions
for i, row in enumerate(test_set):
    print(f"Expected={row[-1]}, Predicted={predictions[i]}")
