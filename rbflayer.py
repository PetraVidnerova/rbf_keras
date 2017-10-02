
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform
import numpy as np

class RBFLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.centers = self.add_weight(name='centers', 
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=RandomUniform(0.0,1.0),
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim),
                                     initializer='ones',
                                     trainable=True)
            
        super(RBFLayer, self).build(input_shape)  

    def call(self, x):

        C = self.centers[np.newaxis, :, :]
        X = x[:, np.newaxis, :]

        diffnorm = K.sum((C-X)**2, axis=-1)
        ret = K.exp( - self.betas * diffnorm)
        return ret 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
