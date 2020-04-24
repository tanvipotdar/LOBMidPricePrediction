import glob
import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.stats import zscore
from tensorflow import keras
from sklearn.metrics import classification_report
import random

# path where data is stored
STOCK_PATH = "{}".format(stock)
DATA_PATH = "/content/data/"
PATH = DATA_PATH+STOCK_PATH

# column names
levels = 5
nums = map(str, range(1,levels+1))
col_names = [y + x for x in nums for y in ['ask_price_', 'ask_size_', 'bid_price_', 'bid_size_']]

# train-test split
days = list(range(62))
train_days = days[:50]
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
  S = []
  for i in range(len(T)-1):
    if T[i+1]-1 - T[i] > 0:
      tilda_t = random.sample(range(T[i], T[i+1]-1), 1)[0]
    else:
      tilda_t = T[i]
    S.append(tilda_t)
  sampled_data = data.iloc[S].queue_imbalance.values.tolist()
  filtered_data.queue_imbalance = [np.nan] + sampled_data
  return filtered_data


def process_data(data):
    data['queue_imbalance'] = (data.ask_size_1-data.bid_size_1)/(data.ask_size_1+data.bid_size_1)
    data['midprice'] = (data.ask_price_1+data.bid_price_1)/2
    data['change'] = data.midprice - data.midprice.shift()  
    filtered_data = sample_imbalance(data)
    filtered_data['label'] = np.where(filtered_data.change > 0, 1, 0)
    normalised_data = normalise_data(filtered_data)
    return normalised_data.iloc[:,[-4]].values, normalised_data.iloc[:,[-1]].values
    

train_data = read_data(train_days, 'Training')
test_data = read_data(test_days, 'Test')

X_train, y_train = process_data(train_data)
X_test, y_test = process_data(train_data)

clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('{:.4f}'.format(clf.score(X_test, y_test)))
print(classification_report(y_test, y_pred))