from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization

def generator(phase_train=True, params={'z_size':200, 'strides':(2,2,2), 'kernel_size':(4,4,4)}):
    """
    Returns a Generator Model with input params and phase_train 
    Args:
        phase_train (boolean): training phase or not
        params (dict): Dictionary with model parameters    
    Returns:
        model (keras.Model): Keras Generator model
    """

    z_size = params['z_size']
    strides = params['strides']
    kernel_size = params['kernel_size'] 
    
    inputs = Input(shape=(1, 1, 1, z_size))

    g1 = Deconv3D(filters=512, kernel_size=kernel_size,
                  strides=(1, 1, 1), kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='valid')(inputs)
    g1 = BatchNormalization()(g1, training=phase_train)
    g1 = Activation(activation='relu')(g1)

    g2 = Deconv3D(filters=256, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(g1)
    g2 = BatchNormalization()(g2, training=phase_train)
    g2 = Activation(activation='relu')(g2)

    g3 = Deconv3D(filters=128, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(g2)
    g3 = BatchNormalization()(g3, training=phase_train)
    g3 = Activation(activation='relu')(g3)

    g4 = Deconv3D(filters=64, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(g3)
    g4 = BatchNormalization()(g4, training=phase_train)
    g4 = Activation(activation='relu')(g4)

    g5 = Deconv3D(filters=1, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(g4)
    g5 = BatchNormalization()(g5, training=phase_train)
    g5 = Activation(activation='sigmoid')(g5) 

    model = Model(inputs=inputs, outputs=g5)
    model.summary()

    return model

def discriminator(phase_train = True, params={'cube_len':64, 'strides':(2,2,2), 'kernel_size':(4,4,4), 'leak_value':0.2}):
    """
    Returns a Discriminator Model with input params and phase_train 
    Args:
        phase_train (boolean): training phase or not
        params (dict): Dictionary with model parameters    
    Returns:
        model (keras.Model): Keras Discriminator model
    """
    cube_len = params['cube_len']
    strides = params['strides']
    kernel_size = params['kernel_size'] 
    leak_value = params['leak_value']
    
    inputs = Input(shape=(cube_len, cube_len, cube_len, 1))

    d1 = Conv3D(filters=64, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(inputs)
    d1 = BatchNormalization()(d1, training=phase_train)
    d1 = LeakyReLU(leak_value)(d1)

    d2 = Conv3D(filters=128, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(d1)
    d2 = BatchNormalization()(d2, training=phase_train)
    d2 = LeakyReLU(leak_value)(d2)

    d3 = Conv3D(filters=256, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(d2)
    d3 = BatchNormalization()(d3, training=phase_train)
    d3 = LeakyReLU(leak_value)(d3)

    d4 = Conv3D(filters=512, kernel_size=kernel_size,
                  strides=strides, kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='same')(d3)
    d4 = BatchNormalization()(d4, training=phase_train)
    d4 = LeakyReLU(leak_value)(d4)

    d5 = Conv3D(filters=1, kernel_size=kernel_size,
                  strides=(1, 1, 1), kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='valid')(d4)
    d5 = BatchNormalization()(d5, training=phase_train)
    d5 = Activation(activation='sigmoid')(d5) 

    model = Model(inputs=inputs, outputs=d5)
    model.summary()

    return model