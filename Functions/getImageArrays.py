import numpy as np
import os
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy
import sys
import math
from PIL import Image

def getImageArrays(path, side_length, max_num_images): #returns list of images arrays for a specified path
    image_names = os.listdir(path)
    examples = []
    for image_name in image_names:
        if len(examples) == max_num_images:
            return examples
        if image_name.split(".")[-1] != "DS_Store":
            try:
                cur_image_path = path + image_name
                cur_image = np.array(ndimage.imread(cur_image_path, flatten=False))
                cur_array_resized = scipy.misc.imresize(cur_image, size=(side_length, side_length))
                assert(cur_array_resized.shape == (side_length, side_length, 3))
                examples += [cur_array_resized]
#                cur_array_flattened = cur_array_resized.reshape((side_length*side_length*3)).T
#                examples += [cur_array_flattened]
                # print("Cur array", np.shape(cur_array_resized)) 
            except ValueError:
                print("Error in creating examples", image_name)
            except AssertionError:
                print("Invalid shape", image_name)
            
        # print("Examples shape", path, np.shape(examples))
    return examples

def getExamples(side_length, image_path, test_ratio, max_num_images=-10):
    print("getExamples!")
    cow_images_path = image_path + "cows/"
    notCow_image_path = image_path + "notcows/"

    max_num_examples_cow = max_num_images // 2 
    max_num_examples_notCow = max_num_images - max_num_examples_cow
    examples_cow = getImageArrays(cow_images_path, side_length, max_num_examples_cow)
    print("examplesCow", np.shape(examples_cow))
    print(notCow_image_path, "herererer")
    examples_notCow = getImageArrays(notCow_image_path, side_length, max_num_examples_notCow)
    print("examplesNotCow", np.shape(examples_notCow))
    print("finished getImageArrays!")
    #labels_cow = np.ones(len(examples_cow))
    cow_labels_len, not_cow_labels_len = len(examples_cow), len(examples_notCow)
    labels_cow_data = [np.zeros(shape=(cow_labels_len, 1)), np.ones(shape=(cow_labels_len, 1))]
    labels_not_cow_data = [np.ones(shape=(not_cow_labels_len, 1)), np.zeros(shape=(not_cow_labels_len, 1))]
 
    print("labels_cow_data", labels_cow_data[0].shape)
    
    labels_cow = np.stack(labels_cow_data, axis=1)
    labels_notCow = np.stack(labels_not_cow_data, axis=1)
 
    labels_cow = np.squeeze(labels_cow)
    labels_notCow = np.squeeze(labels_notCow)
    
    print("lables_cow", labels_cow.shape)
    print("lables_notCow", labels_notCow.shape)

    examples = np.concatenate((examples_cow,examples_notCow))
    labels = np.concatenate((labels_cow,labels_notCow))
    print("Examples Before", examples.shape)
    print("labels Before", labels.shape)
    shuffled_indexing = np.random.permutation(labels.shape[0])
    examples = examples[shuffled_indexing]
    labels = labels[shuffled_indexing]
    print("Examples After", examples.shape)
    print("labels After", labels.shape)

    assert(len(examples) == len(labels)), "labels and examples don't match"
    
    #seperate train and test examples
    number_examples_test = int(len(examples)*test_ratio)
    number_labels_test = int(len(labels)*test_ratio)

    examples_test = examples[:number_examples_test]
    examples_train = examples[number_examples_test:]
    labels_test = labels[:number_labels_test]
    labels_train = labels[number_labels_test:]
    #print("Number of training examples: ", examples_train.T.shape[1])
    #print("Number of test examples: ", examples_test.T.shape[1])
    
    #reshape labels and examples for future matrix operations
    labels_train = np.reshape(labels_train,(1,len(labels_train)))
    labels_test = np.reshape(labels_test,(1,len(labels_test)))
    examples_train = examples_train
    examples_test = examples_test

     # Standardize color values of the image (decrease computational cost durring cross entropy)
    standardized_train_examples = examples_train/255 #225 is the maximum rgb value/ This is done to decrease varaince in inputs thus more efficint
    standardized_test_examples = examples_test/255
#    print("Final Shapes:", "test:", standardized_test_examples.shape, "train:", standardized_train_examples.shape)
    return standardized_train_examples, labels_train.T, standardized_test_examples, labels_test.T

# x1,y1,x2,y2 = getExamples(side_length = 150,
# image_path = "../Data/Logistic_Regression_Data/",
# test_ratio = .3)
# print("X train", x1.shape)
# print("Y train", y1.shape)
# print("X test", x2.shape)
# print("Y train", y2.shape)