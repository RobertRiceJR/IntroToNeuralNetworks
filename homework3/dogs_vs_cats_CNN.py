######################################################################
#####                                                            #####
##### CNN for Dogs vs. Cats Project                              #####
#####                                                            #####
######################################################################

######################################################################
##### LIBRARY IMPORTS                                            #####
######################################################################

import os
import random
import warnings

# Components for building the layered CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense

# The ImageDataGenerator will carry out the flow of image data from storage to memory
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Set the base directory for the dataset
# Assume the dataset is present and the folder structure matches what is expected
src = 'Dataset/PetImages/'


######################################################################
##### PARTITION DATASET INTO TRAINING SET AND TESTING SET        #####
######################################################################

# Import the function for splitting the dataset into the training set and the testing set
warnings.filterwarnings("ignore")
from dataset_utilities import train_test_split

# Create separate folders for storing the training samples and the test samples
#if not os.path.isdir(src + 'Train/'):

# Partition the dataset 
train_test_split(src)


######################################################################
##### BUILD AND COMPILE THE CNN                                  #####
######################################################################

# Define hyperparameters
FILTER_SIZE = 3                         # This is the sliding window that will scan an image and create a feature set
NUM_FILTERS = 32                        # We will use a total of 32 filters
INPUT_SIZE  = 32                        # Compress image dimensions to 32 x 32 (may lose some data)
MAXPOOL_SIZE = 2                        # 2 x 2 maxppoling size halves the input dimensions
BATCH_SIZE = 16                         # Use 16 training samples per batch
STEPS_PER_EPOCH = 20000//BATCH_SIZE     # Number of iterations per epoch; the '//' operator divides, but truncates the decimal
EPOCHS = 10                             # Use 10 epochs

# Start with base sequential layer model
model = Sequential()

# Add first 2D convolutional layer; this layer reads the actual image files
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), input_shape = (INPUT_SIZE, INPUT_SIZE, 3), activation = 'relu'))

# Add first subsampling layer using max pooling
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))

# Add second 2D convolutional layer; this layer reads the subsampled feature map from the first convolutional layer
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), activation = 'relu'))

# Add second subsampling layer, also using max pooling
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))

# Flatten the multidmensional vector received from the second subsampling layer into a one-dimensional vector
model.add(Flatten())

# Add a dense fully connected layer with 128 nodes
model.add(Dense(units = 128, activation = 'relu'))

# Add dropout layer to help reduce overfitting
model.add(Dropout(0.5))

# Add second fully connected layer, with only a single node and using a sigmoid activation function
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compile model using adam optimizer and binary cross-entropy loss function
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


######################################################################
##### SET UP TO BATCH LOAD TRAINING AND TESTING SAMPLES          #####
######################################################################

# Create ImageDataGenerators to load batches of images at a time into memory
# Use separate ImageDataGenerators for training and testing
training_data_generator = ImageDataGenerator(rescale = 1./255)
testing_data_generator = ImageDataGenerator(rescale = 1./255)

# Create batch-loaded training set
training_set = training_data_generator.flow_from_directory(src+'Train/',
                                                target_size = (INPUT_SIZE, INPUT_SIZE),
                                                batch_size = BATCH_SIZE,
                                                class_mode = 'binary')

# Create batch-loaded testing set
test_set = testing_data_generator.flow_from_directory(src+'Test/',
                                             target_size = (INPUT_SIZE, INPUT_SIZE),
                                             batch_size = BATCH_SIZE,
                                             class_mode = 'binary')

######################################################################
##### TRAINING                                                   #####
######################################################################

# Train the model
model.fit_generator(training_set, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose=0)


######################################################################
##### TESTING                                                    #####
######################################################################

# Test the model
score = model.evaluate_generator(test_set, steps=100)

######################################################################
##### RESULTS                                                    #####
######################################################################

for idx, metric in enumerate(model.metrics_names):
    print("{}: {}".format(metric, score[idx]))




