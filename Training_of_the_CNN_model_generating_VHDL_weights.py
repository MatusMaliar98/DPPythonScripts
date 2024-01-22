# CODE FOR PREPARING NEURON MODEL, TRAINING IT, AND EXTRACTING RELEVAND DATA IN VHDL FORMAT

import sys


import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

# setting maximum precision for output and maximum log size
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=100)

# downloading training and testing handwritten data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# setting input shape for the model (height, width, 1 for greyscale)
input_shape = (28, 28, 1)
input_shape2 = (13, 13, 3)

# converting data from 0-255 to 0-1 values so model can work with them more efficiently
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train=x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test=x_test/255.0

# setting classifier output as numbers since model is recognizing numbers
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

# number of times model is going to be repeating the training
epochs = 10

# building model for recognition
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(3, (2,2), padding='valid', activation='relu', input_shape=input_shape, use_bias=True),
    tf.keras.layers.MaxPool2D(strides=(2,2), padding='valid'),
    tf.keras.layers.Conv2D(2, (2, 2), padding='valid', activation='relu', use_bias=True),
    tf.keras.layers.MaxPool2D(strides=(2, 2), padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='relu', use_bias=True),
    tf.keras.layers.Dense(10, activation='softmax', use_bias=True)
])

# compiling model
model.compile( loss='categorical_crossentropy', metrics=['acc'])

# actual training of the model
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=epochs,
                    validation_split=0.1)

# this part is just to visualize training in the console output
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training Loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)

# evaluating model accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)

original_stdout = sys.stdout

# testing predicion on number 7
prediction=model.predict(x_test)
print(np.argmax(prediction[3]))
print(prediction[3])

with open('pred-out.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    for height in range(28):
        for width in range(28):
            print(x_test[3][height][width][0])
    sys.stdout = original_stdout

j = 0;
for layer in model.layers:
    with open('file_'+str(j)+'.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        weights = layer.get_weights()
        print(weights)
        sys.stdout = original_stdout
    j = j+1


layer = model.layers[0]
with open('layer0code.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    weights = layer.get_weights()[0]
    weightsStr = weights.astype(str)
    string = ""
    for ftmap in range(3):
        string = string + "CONSTANT FMAP_1_"+str(ftmap+1)+": FILTER_0_TYPE:= ("
        for height in range(2):
            string += "("
            for width in range(2):
                if width == 1:
                    string += "to_sfixed("+weightsStr[height][width][0][ftmap]+",F_POS,F_NEG)"
                else:
                    string += "to_sfixed("+weightsStr[height][width][0][ftmap]+",F_POS,F_NEG),"
            if height == 1:
                string += ")"
            else:
                string += "),\n"
        string += ");\n\n"
    string = string + ");\n\n"

    bias = layer.get_weights()[1]
    biasStr = bias.astype(str)
    for neuron in range(3):
        string = string + "CONSTANT BIAS_VAL_"+str(neuron+1)+": sfixed(F_POS DOWNTO F_NEG) := to_sfixed("+biasStr[neuron]+",F_POS,F_NEG);\n"
    string = string + "\n"
    print(string)
    sys.stdout = original_stdout



layer = model.layers[2]
with open('layer2code.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    weights = layer.get_weights()[0]
    weightsStr = weights.astype(str)
    string = ""
    for ftmap in range(2):
        for inputMap in range(3):
            string = string + "CONSTANT FMAP_3_" + str(ftmap+1) + "_" +str(inputMap+1)+": FILTER_3_TYPE:= ("
            for height in range(2):
                string += "("
                for width in range(2):
                    if width == 1:
                        string += "to_sfixed("+weightsStr[height][width][inputMap][ftmap]+",F_POS,F_NEG)"
                    else:
                        string += "to_sfixed("+weightsStr[height][width][inputMap][ftmap]+",F_POS,F_NEG),"
                if height == 1:
                    string += ")"
                else:
                    string += "),\n"
            string += ");\n\n"
    string = string + ");\n\n"

    bias = layer.get_weights()[1]
    biasStr = bias.astype(str)
    for neuron in range(2):
        string = string + "CONSTANT L3_BIAS_VAL_"+str(neuron+1)+": sfixed(F_POS DOWNTO F_NEG) := to_sfixed("+biasStr[neuron]+",F_POS,F_NEG);\n"
    string = string + "\n"
    print(string)
    sys.stdout = original_stdout




layer = model.layers[5]
with open('layer5code.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    weights = layer.get_weights()[0]
    weightsStr = weights.astype(str)
    string = "constant L5_WEIGHT: L5_WEIGHT_TYPE:=\n(\n"
    for neuron in range(10):
        string = string + "("
        for element in range(72):
            if element == 71:
                string = string + "to_sfixed("+weightsStr[element][neuron]+",F_POS,F_NEG)"
            else:
                string = string + "to_sfixed("+weightsStr[element][neuron]+",F_POS,F_NEG),"
        if neuron == 9:
            string = string + ")\n"
        else:
            string = string + "),\n"
    string = string + ");\n\n"

    bias = layer.get_weights()[1]
    biasStr = bias.astype(str)
    string = string + " CONSTANT L5_BIAS_VAL:L5_BIAS_TYPE:= ("
    for neuron in range(10):
        if neuron == 9:
            string = string + "to_sfixed(" + biasStr[neuron] + ",F_POS,F_NEG)"
        else:
            string = string + "to_sfixed(" + biasStr[neuron] + ",F_POS,F_NEG),"
    string = string + ");"
    print(string)
    sys.stdout = original_stdout



layer = model.layers[6]
with open('layer6code.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    weights = layer.get_weights()[0]
    weightsStr = weights.astype(str)
    string = "constant L6_WEIGHT: L6_WEIGHT_TYPE:=\n(\n"
    for neuron in range(10):
        string = string + "("
        for element in range(10):
            if element == 9:
                string = string + "to_sfixed("+weightsStr[element][neuron]+",F_POS,F_NEG)"
            else:
                string = string + "to_sfixed("+weightsStr[element][neuron]+",F_POS,F_NEG),"
        if neuron == 9:
            string = string + ")\n"
        else:
            string = string + "),\n"
    string = string + ");\n\n"

    bias = layer.get_weights()[1]
    biasStr = bias.astype(str)
    string = string + " CONSTANT L6_BIAS_VAL:L6_BIAS_TYPE:= ("
    for neuron in range(10):
        if neuron == 9:
            string = string + "to_sfixed(" + biasStr[neuron] + ",F_POS,F_NEG)"
        else:
            string = string + "to_sfixed(" + biasStr[neuron] + ",F_POS,F_NEG),"
    string = string + ");"
    print(string)
    sys.stdout = original_stdout



# saving neuron and filter weights into the files so i can transfer them into VHDL code manually (instead of layer 5)
#original_stdout = sys.stdout
#for layer in model.layers:
#    with open('file_'+str(j)+'.txt', 'w') as f:
#        sys.stdout = f  # Change the standard output to the file we created.
#        weights = layer.get_weights()
#        print(weights)
#        sys.stdout = original_stdout
#    j = j+1
