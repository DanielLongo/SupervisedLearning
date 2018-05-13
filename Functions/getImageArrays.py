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
    image_names = image_names[:max_num_images]
    examples = []
    for image_name in image_names:
        if image_name.split(".")[-1] != "DS_Store":
            try:
                cur_image_path = path + image_name
                cur_image = np.array(ndimage.imread(cur_image_path,flatten=False))
                cur_array_resized = scipy.misc.imresize(cur_image,size=(side_length,side_length))
                cur_array_flattened = cur_array_resized.reshape((side_length*side_length*3)).T
                examples += [cur_array_flattened] 
            except ValueError:
                print("Error in creating examples",image_name)
    return examples

def getExamples(side_length, image_path, test_ratio, max_num_images=-1):
    cow_images_path = image_path + "cows/"
    notCow_image_path = image_path + "notcows/"

    examples_cow = getImageArrays(cow_images_path, side_length, max_num_images)
    examples_notCow = getImageArrays(notCow_image_path, side_length, max_num_images)

    labels_cow = np.ones(len(examples_cow))
    labels_notCow = np.zeros(len(examples_notCow))

    examples = np.concatenate((examples_cow,examples_notCow))
    labels = np.concatenate((labels_cow,labels_notCow))

    assert(len(examples) == len(labels)), "labels and examples don't match"
    
    #seperate train and test examples
    number_examples_test = int(len(examples)*test_ratio)
    number_labels_test = int(len(labels)*test_ratio)

    examples_test = examples[:number_examples_test]
    examples_train = examples[number_examples_test:]
    labels_test = labels[:number_labels_test]
    labels_train = labels[number_labels_test:]
    print("Number of training examples: ", examples_train.T.shape[1])
    print("Number of test examples: ", examples_test.T.shape[1])
    
    #reshape labels and examples for future matrix operations
    labels_train = np.reshape(labels_train,(1,len(labels_train)))
    labels_test = np.reshape(labels_test,(1,len(labels_test)))
    examples_train = examples_train.T
    examples_test = examples_test.T

     # Standardize color values of the image (decrease computational cost durring cross entropy)
    standardized_train_examples = examples_train/255 #225 is the maximum rgb value/ This is done to decrease varaince in inputs thus more efficint
    standardized_test_examples = examples_test/255
    print("Final Shapes:", "test:", standardized_test_examples.shape, "train:", standardized_train_examples.shape)
    return standardized_train_examples, labels_train, standardized_test_examples, labels_test

#x1,y1,x2,y2 = getExamples(side_length = 150,
#image_path = "./Logistic_Regression_Data/",
#test_ratio = .3)
#print(x1.shape)