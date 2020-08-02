# Credit card fraud analysis

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras import backend as K

K.clear_session()
plt.style.use('ggplot')

df = pd.read_csv('../creditcard.csv')

""" Data Cleaning/Preparation """
# due skewedness of the data, tactics most be used to ensure the model doesn't overfit to one specific class (in this case, class 0 - which indicates no fraud has occured)
# to do this, we'll select all the data where there is fraud:

fraud = df.loc[df['Class'] == 1]

# this returns a subset of all fraudulent cases, a total of 492 rows. Now let's select another an equivalent size of non-fraudulent cases
clean = df.loc[df['Class'] == 0].iloc[:492]

data = pd.concat([clean, fraud])
# print(f"CONCATENATED VALUES: {pd.value_counts(data['Class'])}")

data = data.sample(frac=1).reset_index(drop=True)
# print(data)
#  Now the "data" dataframe is a randomly shuffled data set containing equal amounts per each class for this problem

# Let's create a testing set and a training set which we can then use to evaluate our model

index = int(data.shape[0] * 0.15)

test_x = data[data.columns[1:-2]].iloc[:index]
test_y = data[data.columns[-1]].iloc[:index]
train_x = data[data.columns[1:-2]].iloc[index:]
train_y = data[data.columns[-1]].iloc[index:]

# next we need to standardize this data. We're using Keras in this example, and typically the models like to be trained on numbers in the range [-1, 1]
# To do this we simply subtract by the mean and divide by the standard deviation.

mean = train_x.mean()
std = train_x.std()
train_x = (train_x - mean)/std
print(train_x.min(), train_x.max())

# train_x.plot.hist(bins = 50, color='b', legend=None)
# plt.show()

# as we can see this standardizes our data to be within the range [-1, 1], however there are some outliers that are still outside of this range. Let's clean these up 
# by removing any data point beyond 3 standard deviations from the mean
# for column in train_x:
#     train_x[column] = train_x[column].loc[train_x[column] >= (mean[column] - std[column]*3)]
#     train_x[column] = train_x[column].loc[train_x[column] <= (mean[column] + std[column]*3)]
# train_x.plot.hist(bins = 50, color='b', legend=None)
# plt.show()

# Now let's apply the same logic to the test data - we don't want to use the test data as a calculation for standardization as this may skew the end results of our model
test_x -= mean
test_x /= std

# for column in train_x:
#     test_x[column] = test_x[column].loc[test_x[column] >= (mean[column] - std[column]*3)]
#     test_x[column] = test_x[column].loc[test_x[column] <= (mean[column] + std[column]*3)]

# test_x.plot.hist(bins = 50, color='b', legend=None)
# plt.show()

""" Model training/development """

""" Part 1: Simple Model"""
# epochs = 14
# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(train_x.shape[1], )))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(8, activation='relu'))
# model.add(layers.Dense(1))

# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# history = model.fit(train_x, train_y, epochs = epochs, batch_size = 64, validation_split = 0.2)
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, epochs + 1)

# plt.plot(epochs, acc, 'b', label='Training Accuracy')
# plt.plot(epochs, val_acc, 'bo', label='Validation Accuracy')
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training vs Validation Accuracy")
# plt.legend()
# plt.show()

# plt.plot(epochs, loss, 'b', label='Training Loss')
# plt.plot(epochs, val_loss, 'bo', label='Validation Loss')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training vs Validation Loss")
# plt.legend()
# plt.show()

""" Part 2: K-Fold Validation """

# Since we are dealing with such few data points, we may be able to achieve higher accuracy using this method
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_x.shape[1], )))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model

k = 4
num_val_samples = len(train_x) // k
num_epochs = 80
all_val_acc_history = []
acc_history = []

for i in range(k):
    print(f"Processing fold #: {i}")
    print(f"RANGE: [{i*num_val_samples}, {(i+1)*num_val_samples}]")
    val_data = train_x[i*num_val_samples : (i+1)*num_val_samples]
    val_targets = train_y[i*num_val_samples : (i+1)*num_val_samples]

    partial_train_data = np.concatenate([train_x[:i*num_val_samples], train_x[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_y[:i*num_val_samples], train_y[(i+1)*num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=64, verbose=0, validation_data=(val_data, val_targets))
    all_val_acc_history.append(history.history['val_acc'])
    acc_history.append(history.history['acc'])

avg_val_acc = [np.mean([x[i] for x in all_val_acc_history]) for i in range(num_epochs)]
avg_acc = [np.mean([x[i] for x in acc_history]) for i in range(num_epochs)]
def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_val_acc = smooth_curve(avg_val_acc)
smooth_acc = smooth_curve(avg_acc)
plt.plot(range(1, len(smooth_val_acc) +1), smooth_val_acc, label='Validation Accuracy')   
plt.plot(range(1, len(smooth_acc) +1), smooth_acc, label='Training Accuracy')  
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# results = model.evaluate(test_x, test_y)
# print(f"LOSS: {results[0]*100:.2f}%\nACCURACY: {results[1]*100:.2f}%")