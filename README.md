# rbf_keras
Petra Vidnerová, The Czech Academy of Sciences, Institute of Computer Science

RBF layer for [Keras](https://keras.io/)

You need rbflayer.py to use RBF layers in keras. See test.py for
**very simple** example.

Feel free to use or modify the code. 

## Requirements:
 Keras, Tensorflow, Scikit-learn, optionally Matplotlib (only for test.py)

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

 Because you have created Keras model with a custom layer, you need to take it into 
 account if you need to save it to file and load it.
 Saving is no problem:
 ```
 model.save("some_fency_file_name.h5")
 ```
 but while loading you have to specify your custom object RBFLayer:
 ```
 rbfnet = load_model("some_fency_file_name.h5", custom_objects={'RBFLayer': RBFLayer})
 ```

## See also:
**Issue [#1](https://github.com/PetraVidnerova/rbf_keras/issues/1)**:
For hint how to implement different radii for different dimensions.

## Contact:
If you need help, do not hesitate to contact me via petra@cs.cas.cz or write an Issue.

## Acknowledgement: 
This work  was partially supported by the Czech Grant Agency grant 18-23827S 
and institutional support of the Institute of Computer Science RVO 67985807.

