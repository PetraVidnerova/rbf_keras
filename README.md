# rbf_keras
RBF layer for Keras

You need rbflayer.py to use RBF layers in keras. See test.py for
very simple example.

Feel free to use or modify the code. 

## Usage:

```
  # creating RBF network
  rbflayer = RBFLayer(10,
                      initializer=InitCentersRandom(X),
                      betas=2.0,
                      input_shape=(num_inputs,))

  model = Sequential()
  model.add(rbflayer)
  model.add(Dense(n_outputs))
``` 

or using KMeans clustering for RBF centers 

```
  # creating RBFLayer with centers found by KMeans clustering
  rbflayer = RBFLayer(10,
                      initializer=InitCentersKMeans(X),
                      betas=2.0,
                      input_shape=(num_inputs,))
``` 



Hint how to implement different radii for different dimensions: [#1](https://github.com/PetraVidnerova/rbf_keras/issues/1)
