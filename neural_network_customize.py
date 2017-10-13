import numpy
import pandas
import math
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras import utils
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler

numpy.random.seed(7)

dataset = pandas.read_csv('international-airline-passengers.csv', engine='python', skipfooter=3)


# plt.plot(dataset)
# plt.show()

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:i + look_back, :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1])
    return numpy.array(dataX), numpy.array(dataY)


def extract_month(dataset):
    dataset[:] = [int(dataset[x].split('-')[1]) - 1 for x in range(len(dataset))]
    return utils.to_categorical(dataset[:])


dataset = dataset.values

dataset_ticket = dataset[:, 1].astype('float32')
dataset_ticket = numpy.reshape(dataset_ticket, (dataset_ticket.shape[0], 1))
scaler = MinMaxScaler(feature_range=(0, 1))
normalize_dataset = scaler.fit_transform(dataset_ticket)
month_dataset = extract_month(dataset[:,0])
dataset = numpy.concatenate((month_dataset, normalize_dataset), axis=1)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:, :]
print(train.shape)
print(test.shape)

look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

print('Shape trainX ', trainX.shape, '\nShape trainY ', trainY.shape)
print('Shape testX ', testX.shape, '\nShape testY ', testY.shape)

batch_size = 1
time_steps = trainX.shape[1]
features = trainX.shape[2]

model = Sequential()
model.add(LSTM(16, batch_input_shape=(batch_size, time_steps, features), stateful=True, return_sequences=True, dropout=0.0))
model.add(LSTM(16, batch_input_shape=(batch_size, time_steps, features), stateful=True, dropout=0.0))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

for i in range(100):
    model.fit(trainX, trainY, epochs=3, batch_size=batch_size, verbose=2, shuffle=True)
    model.reset_states()

trainPredict = model.predict(trainX, batch_size=batch_size)
testPredict = model.predict(testX, batch_size=batch_size)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

plt.plot(scaler.inverse_transform(normalize_dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
