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
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

p16 = pd.read_csv("../input/properties_2016.csv")
p17 = pd.read_csv("../input/properties_2017.csv")

t16 = pd.read_csv("../input/train_2016_v2.csv")
t17 = pd.read_csv("../input/train_2017.csv")

output = pd.read_csv("../input/sample_submission.csv")

#p = pd.concat([p16, p17])
p = p16
t = pd.concat([t16, t17])

def convert_to_right_month(date_in_some_format):
    return datetime.strptime(date_in_some_format, '%Y-%m-%d').strftime('%m')

def convert_to_right_year(date_in_some_format):
    return datetime.strptime(date_in_some_format, '%Y-%m-%d').strftime('%y')

t['month'] = t['transactiondate'].apply(convert_to_right_month)
t['year'] = t['transactiondate'].apply(convert_to_right_year)

t = t[['parcelid', 'month', 'year', 'logerror']]
#t['transactiondate'] = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')


print(t.head())
print(p.head())

datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')

m1 = t['logerror'].groupby(t['parcelid']).mean()
meanErrorByParcel = pd.DataFrame(m1).reset_index()
m2 = t['logerror'].groupby(t['month']).mean()
meanErrorByDate = pd.DataFrame(m2).reset_index()
m3 = t['logerror'].groupby(t['year']).mean()
meanErrorByYear = pd.DataFrame(m3).reset_index()
#transactiondate
#t['logerror'].groupby(t['transactiondate']).mean()
#def GetUniqueIndex(df, index)
#for i, v in enumerate(p['assessmentyear'].unique()):
#    print(i, v)

print(meanErrorByParcel.head())
print(meanErrorByDate.head())

col = list(p.columns.values)

print(col)

#to preprocess
# 

#22 has tuborspa true or NAN
#32 propertycountylandusecode 
#34 propertyzoningdesc
#49 fireplaceflag Nan True
# 55 taxdelinquencyflag Y Nan

#p[col[22]] = 

def preprocess(df, col):
    kvPair = {}
    for i, v in enumerate(df[col].unique()):
        #print(i, v)
        kvPair[v] = i
        
    df[col] = df[col].map(kvPair)
    
preprocess(p, col[22])
preprocess(p, col[32])
preprocess(p, col[34])
preprocess(p, col[49])
preprocess(p, col[55])

p = p.fillna(0)

print(p.head())

print(list(meanErrorByParcel.columns.values))
traindf = pd.merge(p, t, on='parcelid', right_index=True)

print(p.head())

print(traindf.head())

input1 = traindf.as_matrix().astype('float64')




#print(testDf)
train = input1[:, 1:-1]
print(train.shape)
test = input1[:, -1:]
print(test.shape)
inputScaler = MinMaxScaler(feature_range=(0, 1))
trainX = inputScaler.fit_transform(train)



#print(trainX)
#print(trainY)

print(test)

outputScaler = MinMaxScaler(feature_range=(0, 1))
trainY = outputScaler.fit_transform(test)
#trainY = test

print(trainX.shape)
print(trainY.shape)


model = Sequential()
model.add(Dense(300, input_dim=59, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1000, verbose=2)


#print(pred)


#print(p)

print(p.shape)



print(meanErrorByDate)
print(meanErrorByYear)
m10 = meanErrorByDate[meanErrorByDate['month'] == '10']['logerror']
m11 = meanErrorByDate[meanErrorByDate['month'] == '11']['logerror']
m12 = meanErrorByDate[meanErrorByDate['month'] == '12']['logerror']

y16 = meanErrorByYear[meanErrorByYear['year'] == '16']['logerror']
y17 = meanErrorByYear[meanErrorByYear['year'] == '17']['logerror']

p10 = float(1) + float((float(m10) - float(m10)) / float(m10))
p11 = float(1) + float((float(m11) - float(m10)) / float(m10))
p12 = float(1) + float((float(m12) - float(m10)) / float(m10))

p17 = float(1) + float((float(y17) - float(y16)) / float(y16))

print([p10, p11, p12, p17])

years = [2016, 2017]
months = [10, 11, 12]

d = []
i = 0
#for parcel in output1:
    #d.append({'parcelid': int(parcel[0]), '201610': float(p[i]) + float(m10) + float(y16), '201611': float(p[i]) + float(m11) + float(y16), '201612': float(p[i]) + float(m12) + float(y16), '201710': float(p[i]) + float(m10) + float(y17), '201711': float(p[i]) + float(m11) + float(y17), '201712': float(p[i]) + float(m12) + float(y17)})
    #i = i + 1


output['parcelid'] = output['ParcelId']
testDf = output.merge(p, on='parcelid', how='left')
input2 = testDf.as_matrix().astype('float64')
output1 = input2[:, :1]
result = output1
testDf['month'] = 10
testDf['year'] = 2016
input2 = testDf.as_matrix().astype('float64')
test1 = input2[:, 8:]   
testX = inputScaler.fit_transform(test1)

#print(result.shape)

pred = model.predict(testX)
p = outputScaler.inverse_transform(pred)

#for year in years:
    #for month in months:
        
        #print(result.shape)
        #print(p.shape)
result = np.append(result, p, 1)

p1 = p * p11    
result = np.append(result, p1, 1)

p1 = p * p12
result = np.append(result, p1, 1)

p1 = p * p17
result = np.append(result, p1, 1)

p1 = p * p11 * p17
result = np.append(result, p1, 1)

p1 = p * p12 * p17
result = np.append(result, p1, 1)

output5 = pd.DataFrame(result, columns=['ParcelId', '201610', '201611', '201612', '201710', '201711', '201712'])

#output5['201610'] = output5['201610'] * p10
#output5['201611'] = output5['201611'] * p11
#output5['201612'] = output5['201612'] * p12

#output5['201710'] = output5['201710'] * p17
#output5['201711'] = output5['201711'] * p11 * p17
#output5['201712'] = output5['201712'] * p12 * p17


#cols = output.columns.tolist()
#cols = cols[-1:] + cols[:-1]
#output = output[cols]

print(output5.shape)

print(output5.head())

output5.to_csv('submission.csv')
