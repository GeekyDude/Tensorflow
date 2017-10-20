# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.wrappers.scikit_learn import KerasClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

trainDf = pd.read_csv("../input/train.csv")
testDf = pd.read_csv("../input/test.csv")
sampleDf = pd.read_csv("../input/sample_submission.csv")

print(trainDf.head())
print(testDf.head())
print(sampleDf.head())

nonNaTrainDf = trainDf.fillna(-1)
nonNaTestDf = testDf.fillna(-1)

trainInput = nonNaTrainDf.as_matrix().astype('float64')
testInput = nonNaTestDf.as_matrix().astype('float64')

trainXUS = trainInput[:, 2:]
print(trainXUS.shape)
trainYUS = trainInput[:, 1:2]
print(trainYUS.shape)

inputScaler = MinMaxScaler(feature_range=(0, 1))
trainX = inputScaler.fit_transform(trainXUS)

outputScaler = MinMaxScaler(feature_range=(0, 1))
trainY = outputScaler.fit_transform(trainYUS)

testXUS = testInput[:, 1:]
testX = inputScaler.fit_transform(testXUS)

#model = Sequential()
#model.add(Dense(500, input_dim=57, activation='tanh'))
#model.add(Dense(300, activation='tanh'))
#model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam')

model = Sequential()
model.add(
    Dense(
        200,
        input_dim=57,
        kernel_initializer='glorot_normal',
        ))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(100, kernel_initializer='glorot_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(50, kernel_initializer='glorot_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.15))
model.add(Dense(25, kernel_initializer='glorot_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', metrics = ['accuracy'], loss='binary_crossentropy')

model.fit(trainX, trainY, epochs=100, batch_size=1000, verbose=2)

pred = model.predict(testX)
p = outputScaler.inverse_transform(pred)

output1 = testInput[:, :1]
result = np.append(output1, p, 1)
output5 = pd.DataFrame(result, columns=['id', 'target'])
output5['id'] = output5['id'].astype(int)
output5['target'] = np.where(output5['target']<=0, 0.0, output5['target'])
output5['target'] = np.where(output5['target']>=1, 1.0, output5['target'])
output5.to_csv('submission.csv',mode = 'w', index=False)
