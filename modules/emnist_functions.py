import string
from termcolor import colored

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.metrics import ConfusionMatrixDisplay

###############################################
###############################################
def getEMNIST(datasetName, datasetPath):
    
    import idx2numpy
    fileName = datasetPath+"/"+datasetName+'-images-idx3-ubyte'
    features = idx2numpy.convert_from_file(fileName)
    features = features/tf.math.reduce_max(features)

    fileName = datasetPath+"/"+datasetName+'-labels-idx1-ubyte'
    labels = idx2numpy.convert_from_file(fileName)
    
    return (features, labels)
###############################################
###############################################  
def plotMNIST(x, y, y_pred): 
    indices = np.random.default_rng().integers(0, len(x), (4))
    fig, axes = plt.subplots(2, 2, figsize=(4.5,4.5))

    letters_lower = list(string.ascii_lowercase)
    letters_upper = list(string.ascii_uppercase)

    for index, axis in zip(indices, axes.flatten()): 
        title = "{}/{}".format(y[index],y_pred[index])
        if y[index]>9 and y[index]-10<len(letters_upper):
            title = "{}/{}".format(letters_upper[y[index]-10],letters_upper[y_pred[index]-10])
        axis.imshow(tf.transpose(x[index]), cmap=plt.get_cmap('Reds'), label="A")
        axis.set_title(title)

    axes[0,1].legend(bbox_to_anchor=(1.5,1), loc='upper left', title="Label: True/Predicted") 
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.4, hspace=0.5)
###############################################
###############################################
def plotMNIST_CM(y,y_pred, title):

    fig, axis = plt.subplots(1, 1, figsize=(15,15))

    digits = [str(item) for item in range(0,10)]
    letters_lower = list(string.ascii_lowercase)
    labels = digits + letters_lower

    ConfusionMatrixDisplay.from_predictions(y, y_pred, normalize="true", 
                                        values_format=".2f",
                                        include_values = False,
                                        ax=axis, display_labels=labels)
    axis.set_title(title)
###############################################
###############################################
def getModel(inputShape, nNeurons, hiddenActivation="relu", outputActivation="linear", nOutputNeurons=1):
   
    inputs = tf.keras.Input(shape=inputShape, name="features")
    x = tf.keras.layers.Flatten()(inputs)
    
    for iLayer, n in enumerate(nNeurons):
        x = tf.keras.layers.Dense(n, activation=hiddenActivation, 
                                  bias_initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1),
                                  kernel_initializer='HeNormal',
                                  kernel_regularizer=tf.keras.regularizers.L2(l2=0.01),
                                  name="layer_"+str(iLayer))(x)
                
    outputs = tf.keras.layers.Dense(nOutputNeurons, activation=outputActivation, name = "output")(x)   
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DNN")
    return model
###############################################
###############################################   
def encodeMessage(text, features, labels):

    digits = [str(item) for item in range(0,10)]
    letters_lower = list(string.ascii_lowercase)
    digits_letters = np.array(digits + letters_lower)
        
    fig, axes = plt.subplots(1, len(text), figsize=(10,50))
    if len(text)==1:
        axes = [axes]
    encoded = np.array([[]])
    for axis, char in zip(axes, list(text)):
        char_index = np.argmax(digits_letters==char)
        mask = labels==char_index
        index_in_mask = np.random.default_rng().integers(0, np.sum(mask), (1))
        element_index = np.argwhere(mask>0)[index_in_mask].flatten()[0]
        data = features[element_index]
        if char==" ":
            data = np.full((28,28),0)
        encoded = np.append(encoded, data)
        axis.imshow(tf.transpose(data), cmap=plt.get_cmap('Reds'))
        axis.xaxis.set_major_locator(ticker.NullLocator())
        axis.yaxis.set_major_locator(ticker.NullLocator())
        
    return encoded.reshape( (-1,28,28))
###############################################
############################################### 
