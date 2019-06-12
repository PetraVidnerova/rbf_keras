import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from rbflayer import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt


def load_data():

    data = np.loadtxt("data/data.txt")
    X = data[:, :-1]
    y = data[:, -1:]
    return X, y


if __name__ == "__main__":

    X, y = load_data()

    model = Sequential()
    rbflayer = RBFLayer(10,
                        initializer=InitCentersRandom(X),
                        betas=2.0,
                        input_shape=(1,))
    model.add(rbflayer)
    model.add(Dense(1))

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop())

    model.fit(X, y,
              batch_size=50,
              epochs=2000,
              verbose=1)

    y_pred = model.predict(X)

    print(rbflayer.get_weights())

    plt.plot(X, y_pred)
    plt.plot(X, y)
    plt.plot([-1, 1], [0, 0], color='black')
    plt.xlim([-1, 1])

    centers = rbflayer.get_weights()[0]
    widths = rbflayer.get_weights()[1]
    plt.scatter(centers, np.zeros(len(centers)), s=20*widths)

    plt.show()
