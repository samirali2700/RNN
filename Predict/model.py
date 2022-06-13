import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt


from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

from pickle import dump


df = pd.read_csv('../Data/EURUSD.csv')
df_test = pd.read_csv('../Data/GBPUSD.csv')
df.dropna(inplace=True)
df_test.dropna(inplace=True)

cols = list(df)[1:6]

train_data = df[cols].astype(float)
test_data = df_test[cols].astype(float)


def prepareData(data, steps=1):
    trainX, trainY = [], []
    for i in range(look_back, (len(data) - look_ahead), steps):
        trainX.append(data[(i - look_back): i, 0: data.shape[1]])
        diff = data[(i + look_ahead), 0] - data[i, 0]
        if diff > 0:
            trainY.append(1)
        else:
            trainY.append(0)
    return np.array(trainX), np.array(trainY)


def train(X_train, y_train, X_test, y_test, epochs=50, batch_size=0):
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss',
                         mode='min', verbose=1, save_best_only=True)
    model = Sequential()
    model.add(LSTM(units=UNITS, dropout=DROP_RATE, input_shape=(
        X_train.shape[1], X_train.shape[2]),
        return_sequences=True))
    model.add(LSTM(units=UNITS, dropout=DROP_RATE,
                   return_sequences=True))
    model.add(LSTM(units=UNITS, dropout=DROP_RATE,
                   return_sequences=True))
    model.add(LSTM(units=UNITS,
                   return_sequences=False))

    # output layer
    model.add(Dense(units=1, activation='sigmoid'))

    adam = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, verbose=1, validation_data=(X_test, y_test), callbacks=[mc])

    return history


def plotScore(history):
    epochs = np.arange(0, len(history.history['loss']))
    (fig, axs) = plt.subplots(2, 1)

    train_plot = [['accuracy', 'blue', 'train accuracy'],
                  ['loss', 'red', 'train loss']]
    validation_plot = [['val_accuracy', 'green', 'validation accuracy'],
                       ['val_loss', 'orange', 'validation loss']]

    for i in range(0, 2):
        axs[i].plot(epochs, history.history[str(train_plot[i][0])],
                    color=str(train_plot[i][1]), label=train_plot[i][2])
        axs[i].plot(epochs, history.history[str(validation_plot[i][0])],
                    color=validation_plot[i][1], label=validation_plot[i][2])
        axs[i].grid()
        axs[i].legend()

    axs[0].set(xlabel='epochs', ylabel='Accuracy',
               title='Model train vs validation accuracy')
    axs[1].set(xlabel='epochs', ylabel='Loss',
               title='Model train vs validation Loss')

    plt.tight_layout()
    plt.show()


# scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(0, 1))

scaler = scaler.fit(train_data)
dump(scaler, open('./utils/scaler.pkl', 'wb'))

train_data_scaled = scaler.transform(train_data)
test_data_scaled = scaler.transform(test_data)


X_train = []
y_train = []

STEPS = 90
look_ahead = 30
look_back = 180

X_train, y_train = prepareData(train_data_scaled, STEPS)
X_test, y_test = prepareData(test_data_scaled, STEPS)


#32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
# 256, 300, 512
# 4416 / 2208 / 1104

BATCH_SIZE = 0
DROP_RATE = 0.2
UNITS = 2
LEARNING_RATE = 0.001
EPOCHS = 200


print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('#########')
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)


history = train(X_train, y_train, X_test, y_test, EPOCHS)


def getMinMax(data):
    return min(data), max(data)


min_loss, max_loss = getMinMax(history.history['val_loss'])


with open('log.csv', 'a') as log:
    log.write("\nmin_loss: "+str(min_loss)+" \nmax_loss: "+str(max_loss)+"\nParams: "+str(look_back) +
              "/"+str(look_ahead)+"/"+str(BATCH_SIZE)+"/"+str(DROP_RATE)+"/"+str(UNITS)+"/"+str(LEARNING_RATE)+"/"+str(STEPS)+"/"+str(EPOCHS))


plotScore(history=history)
