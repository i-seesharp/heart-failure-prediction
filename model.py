
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout


def scaling_variables(data, index_arr):
    for i in range(len(index_arr)):
        mean = np.mean(data[:, index_arr[i]])
        std = np.std(data[:, index_arr[i]])
        
        data[:, index_arr[i]] = (data[:, index_arr[i]] - mean)/std
    
    return data


#Import data
dataset = pd.read_csv("heart-failure-dataset.csv")
attributes = np.array(dataset.iloc[:,:-1])
labels = np.array(dataset.iloc[:,-1])


scaled_attributes = scaling_variables(attributes,[0,2,4,6,8,11])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_attributes, labels, test_size=0.2)

class Net(Model):
    def __init__(self):
        super(Net, self).__init__()
        self.dense1 = Dense(input_dim=(None, 12), units=16, activation="relu")
        self.dense2 = Dense(units=8, activation="relu")
        self.dense3 = Dense(units=1, activation="sigmoid")
        self.drop2 = Dropout(0.05)
        
    def call(self, x):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        return out
    
model = Net()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(1e-3)

train_loss = tf.keras.metrics.Mean("train_loss")
train_accuracy = tf.keras.metrics.BinaryAccuracy("train_accuracy")
test_loss = tf.keras.metrics.Mean("test_loss")
test_accuracy = tf.keras.metrics.BinaryAccuracy("test_accuracy")

@tf.function
def train_step(x, labels):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(x, labels):
    predictions = model(x, training=False)
    loss = loss_object(labels, predictions)
    
    
    test_loss(loss)
    test_accuracy(labels, predictions)
    
    
EPOCHS = 500
BATCH_SIZE = 32
NUM_BATCHES = np.size(X_train, axis=0) // BATCH_SIZE
NUM_TEST_BATCHES = np.size(X_test, axis=0) // BATCH_SIZE

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    
    for i in range(NUM_BATCHES):
        X_batch = X_train[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        y_batch = y_train[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        train_step(X_batch, y_batch)
    #The last set which couldnt fit in BATCH_SIZE
    last_X = X_train[NUM_BATCHES*BATCH_SIZE : ]
    last_Y = y_train[NUM_BATCHES*BATCH_SIZE : ]
    
    train_step(last_X, last_Y)
    
    for i in range(NUM_TEST_BATCHES):
        X_batch = X_test[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        y_batch = y_test[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        test_step(X_batch, y_batch)
    #The last set which couldnt fit in BATCH_SIZE
    last_X = X_test[NUM_TEST_BATCHES*BATCH_SIZE : ]
    last_Y = y_test[NUM_TEST_BATCHES*BATCH_SIZE : ]
    
    test_step(last_X, last_Y)
    
    template = " Epoch : {} ,Train Loss : {}, Train Accuracy : {}, Test Loss : {}, Test Accuracy : {}"
    print(template.format(
        epoch + 1,
        train_loss.result(),
        train_accuracy.result(),
        test_loss.result(),
        test_accuracy.result()
        ))
    
temp_test_predictions = model(X_test).numpy().reshape((-1,1))
test_labels = y_test[:].reshape((-1,1))
side_by_side_results = np.concatenate((temp_test_predictions, test_labels), axis=1)


    
    
    
    
        






    


