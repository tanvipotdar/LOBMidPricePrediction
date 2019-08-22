'''Module to create a CNN-LSTM midprice classifier to predict the LOB mid-price direction'''

import numpy as np
import pandas as pd
from tensorflow import keras
from scipy.stats import zscore
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def get_data(path):
    orderbook = pd.read_csv(path, header=None)
    col_names = ['ask_price_', 'ask_size_', 'bid_price_', 'bid_size_']
    nums = map(str, range(1,11))
    orderbook.columns = [y + x for x in nums for y in col_names]
    return orderbook


def normalise_data(data):
    normalised_data = data.apply(zscore)
    return normalised_data


def smooth_midprice_using_k_lookahead(k, normalised_data):
    normalised_data['midprice'] = (normalised_data.ask_price_1+normalised_data.bid_price_1)/2
    # mean of previous k mid-prices
    normalised_data['m_minus'] = normalised_data['midprice'].rolling(window=k).mean()
    # mean of next k mid-prices
    normalised_data['m_plus'] = normalised_data['midprice'][::-1].rolling(window=k).mean()[::-1]
    return normalised_data


def create_midprice_labels(normalised_data):
    alpha = 0.0001
    normalised_data['change'] = (normalised_data.m_plus - normalised_data.m_minus)/normalised_data.m_minus
    # assign categories up, down, stationary
    normalised_data['label'] = pd.cut(normalised_data.change, bins=[-np.inf, -alpha, alpha, np.inf], 
                                    labels=['down', 'stationary', 'up'])
    # drop all unlabelled values (will be first and last k values as they have no m_minus/m_plus value)
    normalised_data.dropna(inplace=True)
    return normalised_data


def reshape_and_categorise_data(normalised_data, N):
    data = normalised_data[:N]
    cols = data.columns.to_list()[:40]
    input_data = data[cols]
    input_array = input_data.to_numpy().reshape(N/100,100,40,1)

    output_data = data.label.to_numpy()[::-100][::-1]
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = output_data.reshape(len(output_data), 1)
    output_array = onehot_encoder.fit_transform(integer_encoded)
    X_train, X_test, y_train, y_test = train_test_split(input_array, output_array, shuffle=False)
    return X_train, X_test, y_train, y_test


def create_model():
    # convolutional layers
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(1,2), input_shape=(100,40,1), strides=(1, 2)))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(1,2), strides=(1, 2)))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(1,10), input_shape=(100,10,1)))
    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))

    # lstm layer
    model.add(keras.layers.LSTM(64))
    model.add(keras.layers.Dense(3,activation='softmax'))

    # compile model and summarize
    adam = keras.optimizers.Adam(lr=0.01, epsilon=1)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def fit_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    BATCH_SIZE = 100
    EPOCHS = 100
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    _, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    return accuracy*100


def run(PATH, k):
    data = get_data(path=PATH)
    N = len(data) // 100
    
    normalised_data  = normalise_data(data)
    data_with_midprice = smooth_midprice_using_k_lookahead(k, normalised_data)
    labeled_data = create_midprice_labels(data_with_midprice)
    X_train, X_test, y_train, y_test = reshape_and_categorise_data(normalised_data, N)
    model = create_model()
    accuracy = fit_and_evaluate_model(model, X_train, X_test, y_train, y_test)
    print("Accuracy is {} %".format(accuracy))
 
