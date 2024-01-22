# NORMALIZING DATA FOR MODEL TRAINING

import sys
import numpy as np
from PIL import Image
import os


picInputHeight = 28
picInputWidth = 28

x_train = np.empty([1500,picInputHeight,picInputWidth],dtype=int)
y_train = np.empty([1500],dtype=int)
x_test = np.empty([940,picInputHeight,picInputWidth],dtype=int)
y_test = np.empty([940],dtype=int)


counter = 0;
for filename in os.listdir('c:\\Users\\I532479\\Documents\\cnnvhdl\\datasetcelebs\\pythonModel\\x_train\\'):
    f = os.path.join('c:\\Users\\I532479\\Documents\\cnnvhdl\\datasetcelebs\\pythonModel\\x_train\\', filename)
    # checking if it is a file
    if os.path.isfile(f):
        if filename.startswith('gates'):
            y_train[counter] = 0
        if filename.startswith('trump'):
            y_train[counter] = 1
        if filename.startswith('zuckerberg'):
            y_train[counter] = 2
        image = Image.open(f).resize((picInputHeight,picInputWidth)).convert('L')
        imageArray = np.array(image)
        x_train[counter] = imageArray
        counter = counter + 1

counter = 0
for filename in os.listdir('c:\\Users\\I532479\\Documents\\cnnvhdl\\datasetcelebs\\pythonModel\\x_test\\'):
    f = os.path.join('c:\\Users\\I532479\\Documents\\cnnvhdl\\datasetcelebs\\pythonModel\\x_test\\', filename)
    # checking if it is a file
    if os.path.isfile(f):
        if filename.startswith('gates'):
            y_test[counter] = 0
        if filename.startswith('trump'):
            y_test[counter] = 1
        if filename.startswith('zuckerberg'):
            y_test[counter] = 2

        image = Image.open(f).resize((picInputHeight,picInputWidth)).convert('L')
        imageArray = np.array(image)
        x_test[counter] = imageArray
        counter = counter + 1

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train=x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test=x_test/255.0

original_stdout = sys.stdout
print(y_test[596])
with open('pred-out.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    for height in range(28):
        for width in range(28):
            print(x_test[596][height][width][0])
    sys.stdout = original_stdout