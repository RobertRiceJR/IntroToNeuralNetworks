######################################################################
##### CNN for Dogs vs. Cats Project                              #####
#####                                                            #####
##### DATASET UTILITIES                                          #####
#####                                                            #####
######################################################################

######################################################################
##### LIBRARY IMPORTS                                            #####
######################################################################

import os				# For functions that depend on the operating system
import shutil			# Allows high-level file operations such as removal of files an folder hierarchical trees
import random			# For random numbers
import piexif			# For dealing with image metadata


######################################################################
##### FUNCTION TRAIN_TEST_SPLIT                                  #####
#####                                                            #####
##### Splits the dataset, creating the training set and testing  #####
##### set.                                                       #####
##### Also removes files that are corrupt, not an image file, or #####
##### should not be used.                                        #####
######################################################################
def train_test_split(src_folder, train_size = 0.8):

	# Remove existing training and testing subfolders
	shutil.rmtree(src_folder + 'Train/Cat/', ignore_errors=True)
	shutil.rmtree(src_folder + 'Train/Dog/', ignore_errors=True)
	shutil.rmtree(src_folder + 'Test/Cat/', ignore_errors=True)
	shutil.rmtree(src_folder + 'Test/Dog/', ignore_errors=True)

	# Create new empty train and test folders
	os.makedirs(src_folder + 'Train/Cat/')
	os.makedirs(src_folder + 'Train/Dog/')
	os.makedirs(src_folder + 'Test/Cat/')
	os.makedirs(src_folder + 'Test/Dog/')

	# Retrieve the cat images
	_, _, cat_images = next(os.walk(src_folder + 'Cat/'))

	# These files are corrupt or they are non-image files	
	files_to_be_removed = ['Thumbs.db', '666.jpg', '835.jpg']
	
	# Remove unwanted files
	for file in files_to_be_removed:
		cat_images.remove(file)
	
	# Calculate number of cat images left, and determine the number
	# of cat images to use for the training and testing sets
	num_cat_images = len(cat_images)
	num_cat_images_train = int(train_size * num_cat_images)
	num_cat_images_test = num_cat_images - num_cat_images_train

	# Now retrieve the dog images
	_, _, dog_images = next(os.walk(src_folder+'Dog/'))

	# These files are corrupt or they are non-image files
	files_to_be_removed = ['Thumbs.db', '11702.jpg']
	
	# Remove unwanted files
	for file in files_to_be_removed:
		dog_images.remove(file)

	# Calculate number of dog images left, and determine the number
	# of dog images to use for the training and testing sets
	num_dog_images = len(dog_images)
	num_dog_images_train = int(train_size * num_dog_images)
	num_dog_images_test = num_dog_images - num_dog_images_train

	# Randomly assign cat images to the training set
	cat_train_images = random.sample(cat_images, num_cat_images_train)

	for img in cat_train_images:
		shutil.copy(src=src_folder + 'Cat/' + img, dst=src_folder + 'Train/Cat/')

	# Place leftover cat images in the testing set
	cat_test_images  = [img for img in cat_images if img not in cat_train_images]
	
	for img in cat_test_images:
		shutil.copy(src=src_folder + 'Cat/' + img, dst=src_folder + 'Test/Cat/')

	# Randomly assign dog images to the training set
	dog_train_images = random.sample(dog_images, num_dog_images_train)

	for img in dog_train_images:
		shutil.copy(src=src_folder + 'Dog/' + img, dst=src_folder + 'Train/Dog/')
	
	# Place leftover dog images in the testing set
	dog_test_images  = [img for img in dog_images if img not in dog_train_images]
	
	for img in dog_test_images:
		shutil.copy(src=src_folder + 'Dog/' + img, dst=src_folder + 'Test/Dog/')

	# Remove corrupted exif data from the dataset
	remove_exif_data(src_folder + 'Train/')
	remove_exif_data(src_folder + 'Test/')


# Helper function to remove corrupt exif data from Microsoft's dataset
def remove_exif_data(src_folder):
	_, _, cat_images = next(os.walk(src_folder + 'Cat/'))

	for img in cat_images:
		try:
			piexif.remove(src_folder + 'Cat/' + img)
		except:
			pass

	_, _, dog_images = next(os.walk(src_folder + 'Dog/'))

	for img in dog_images:
		try:
			piexif.remove(src_folder + 'Dog/' + img)
		except:
			pass



# src_folder = 'Dataset/PetImages/'
# train_test_split(src_folder)