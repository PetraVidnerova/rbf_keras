import random 
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Orthogonal, Constant
import numpy as np

class InitCentersRandom(Initializer):

    def __init__(self, X):
        self.X = X 

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx,:]

        
class RBFLayer(Layer):

    def __init__(self, output_dim, initializer=None, betas=0.01, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas 
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
            #self.initializer = Orthogonal()
        else:
            self.initializer = initializer 
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.centers = self.add_weight(name='centers', 
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim),
                                     initializer=Constant(value=self.init_betas),
                                     #initializer='ones',
                                     trainable=True)
            
        super(RBFLayer, self).build(input_shape)  

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = (C-x.T).T
        return K.exp( -self.betas * K.sum(H**2, axis=1))
        
        #C = self.centers[np.newaxis, :, :]
        #X = x[:, np.newaxis, :]

        #diffnorm = K.sum((C-X)**2, axis=-1)
        #ret = K.exp( - self.betas * diffnorm)
        #return ret 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
