import glob
import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import zscore
from tensorflow import keras
import random

# Constants
# batch size for model
BATCH_SIZE = 64

# number of epochs
EPOCHS = 200

# learning rate and epsilon for ADAM optimizer
LEARNING_RATE = 0.01
EPSILON = 1

# path where data is stored
STOCK_PATH = "{}".format(stock)
DATA_PATH = "/content/data/"
PATH = DATA_PATH+STOCK_PATH

# bool flag to decide if data needs to be written
write_data = True

# column names
levels = 5
nums = map(str, range(1,levels+1))
col_names = [y + x for x in nums for y in ['ask_price_', 'ask_size_', 'bid_price_', 'bid_size_']]

#split into train, test and validation days
days = list(range(62))
train_days = days[:38]
val_days = days[38:50]
test_days = days[50:62]


all_files = glob.glob(PATH + "/*orderbook_5.csv")
all_files.sort()
print("Number of files:",len(all_files))

def read_data(days, data_type):
    dfs = []
    print("Reading {}".format(data_type))
    for filename in [all_files[i] for i in days]:
        df = pd.read_csv(filename, index_col=None, header=None, names=col_names)
        dfs.append(df)
    data = pd.concat(dfs, axis=0, ignore_index=True)
    return data


def normalise_data(data):
  final = data.copy()
  for colname in col_names:
    final[colname] = np.log(final[colname]).diff()
    final[colname] = (final[colname] - final[colname].mean())/final[colname].std()
  final = final.dropna()
  return final

def sample_imbalance(data):
  filtered_data = data[data.change!=0]
  T = filtered_data.index
  S = [0]
  for i in range(len(T)-1):
    if T[i+1]-1 - T[i] > 0:
      tilda_t = random.sample(range(T[i], T[i+1]-1), 1)[0]
    else:
      tilda_t = T[i]
    S.append(tilda_t)
  sampled_data = data.iloc[S]
  sampled_data.change = filtered_data.change.tolist()
  return sampled_data

def filter_data_r(data):
    data['midprice'] = (data.ask_price_1+data.bid_price_1)/2
    data['change'] = data.midprice - data.midprice.shift()
    filtered_data = sample_imbalance(data)
    filtered_data['label'] = np.where(filtered_data.change > 0, 1, 0)
    return filtered_data 

def filter_data(data):
  fd = filter_data_r(data)
  nd = normalise_data(fd)
  return nd

def reshape_and_categorise_data(normalised_data):
    n = len(normalised_data) - len(normalised_data)%100
    data = normalised_data[:n]

    input_data = data[col_names]
    input_array = input_data.to_numpy().reshape(n//100,100,levels*4,1)

    output_data = data.label.to_numpy()[::-100][::-1]
    return input_array, output_data

def preprocess_data(data, data_type):
    data = filter_data(data)
    X, y = reshape_and_categorise_data(data)
    print("{} input shape:".format(data_type), X.shape)
    print("{} output shape:".format(data_type), y.shape)
    return X, y

train_data = read_data(train_days, data_type="Training")
val_data = read_data(val_days, data_type="Validation")
test_data = read_data(test_days, data_type="Test")

X_train, y_train = preprocess_data(train_data, data_type="Training")
X_val, y_val = preprocess_data(val_data, data_type="Validation")
X_test, y_test = preprocess_data(test_data, data_type="Test")


def create_lstm():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(1,2), input_shape=(100,levels*4,1), strides=(1, 2)))
    model.add(keras.layers.LeakyReLU(alpha=0.01))

    model.add(keras.layers.Conv2D(filters=16, kernel_size=(1,2), strides=(1, 2)))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,1)))

    model.add(keras.layers.Conv2D(filters=32, kernel_size=(1,levels), input_shape=(100,levels,1)))
    model.add(keras.layers.LeakyReLU(alpha=0.01)) 

    # lstm layer
    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    model.add(keras.layers.LSTM(64,kernel_regularizer=keras.regularizers.l2(0.01), return_sequences=False))

    model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # compile model and summarize  
    adam = keras.optimizers.Adam(lr=LEARNING_RATE, epsilon=1)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def create_rnn():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(1,2), input_shape=(100,levels*4,1), strides=(1, 2)))
    model.add(keras.layers.LeakyReLU(alpha=0.01))

    model.add(keras.layers.Conv2D(filters=16, kernel_size=(1,2), strides=(1, 2)))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,1)))

    model.add(keras.layers.Conv2D(filters=32, kernel_size=(1,levels), input_shape=(100,levels,1)))
    model.add(keras.layers.LeakyReLU(alpha=0.01)) 

    # lstm layer
    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    model.add(keras.layers.SimpleRNN(64,kernel_regularizer=keras.regularizers.l2(0.01),return_sequences=False))

    model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # compile model and summarize  
    adam = keras.optimizers.Adam(lr=LEARNING_RATE, epsilon=1)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
 
    

model = create_lstm()

callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
print("Batch size:{}, Learning Rate:{}".format(BATCH_SIZE, LEARNING_RATE))
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,  
                    validation_data=(X_val, y_val), verbose=1, callbacks=[callback])
score, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print("Accuracy is {}%".format(accuracy*100))


y_pred = model.predict_classes(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
print(classification_report(y_pred, y_test, digits=4))
