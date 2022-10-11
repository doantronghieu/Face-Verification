# Custom L1 Distance layer module 
# WHY DO WE NEED THIS: its needed to load the custom model

# Import dependencies
import tensorflow as tf; import tensorflow.keras as tfk
from tensorflow.keras.layers import Layer

# Build Distance Layer
# Siamese L1 Distance class
class L1_Dist_Layer(tfk.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__() # Init method - inheritance
    
    # Similarity calculation
    def call(self, embedding_input, embedding_validation): 
        return tf.math.abs(embedding_input - embedding_validation)