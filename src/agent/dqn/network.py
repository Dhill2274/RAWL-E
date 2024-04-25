import tensorflow
from tensorflow import keras
import keras.layers as layers
from keras import initializers as initialiser

class MyNetwork(keras.Model):
    def __init__(self, n_features, hidden_units, n_actions):
        super(MyNetwork, self).__init__()
        self.n_features = n_features
        self.input_layer = layers.InputLayer(input_shape=n_features,)
        self.hidden_layer_1 = layers.Dense(hidden_units, activation="relu", kernel_initializer=initialiser.HeNormal())
        self.hidden_layer_2 = layers.Dense(hidden_units, activation="relu", kernel_initializer=initialiser.HeNormal())
        self.output_layer = layers.Dense(n_actions, activation="linear", kernel_initializer=initialiser.HeNormal())
        
        self.model = keras.Sequential([
            self.input_layer,
            self.hidden_layer_1,
            self.hidden_layer_2,
            self.output_layer
        ])
        
    def call(self, inputs):
        z = self.model(inputs)
        return z