# Credit card fraud analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras import backend as K
from keras import regularizers
from keras import optimizers

K.clear_session()
plt.style.use('ggplot')

def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

def build_model(train_x):
    model = models.Sequential()
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.002), activation='relu', input_shape=(train_x.shape[1], )))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.002), activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.002), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model

def main():
  df = pd.read_csv('creditcard.csv')

  """ Data Cleaning/Preparation """
  # due skewedness of the data, tactics most be used to ensure the model doesn't overfit to one specific class (in this case, class 0 - which indicates no fraud has occured)
  # to do this, we'll select all the data where there is fraud:

  fraud = df.loc[df['Class'] == 1]

  clean = df.loc[df['Class'] == 0].iloc[:5000]

  data = pd.concat([clean, fraud])

  data = data.sample(frac=1).reset_index(drop=True)
  #  Now the "data" dataframe is a randomly shuffled data set containing equal amounts per each class for this problem

  # Let's create a testing set and a training set which we can then use to evaluate our model

  index = int(data.shape[0] * 0.25)

  test_x = data[data.columns[1:-2]].iloc[:index]
  test_y = data[data.columns[-1]].iloc[:index]
  test_x.to_csv("test_x.csv")
  test_y.to_csv("test_y.csv")
  train_x = data[data.columns[1:-2]].iloc[index:]
  train_y = data[data.columns[-1]].iloc[index:]

  # next we need to standardize this data. We're using Keras in this example, and typically the models like to be trained on numbers in the range [-1, 1]
  # To do this we simply subtract by the mean and divide by the standard deviation.

  mean = train_x.mean()
  std = train_x.std()
  train_x = (train_x - mean)/std

  test_x -= mean
  test_x /= std

  """ Model training/development """

  class_weight = {0: 1.,
                1: 15.}

  num_epochs = 80
  model = build_model(train_x)
  model.fit(train_x, train_y, epochs=num_epochs, batch_size=64)

  results = model.evaluate(test_x, test_y)
  print(f"--------------TRAINED MODEL--------------\nLOSS: {results[0]:.2f}\nACCURACY: {results[1]*100:.2f}%")

if __name__ == "__main__":
  main()