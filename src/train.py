import os
import numpy as np
import dataIO as d

from keras.models import Sequential
from keras.optimizers import Adam
from models import discriminator as disc_model
from models import generator as gen_model

n_epochs   = 10000
batch_size = 30
g_lr       = 0.008
d_lr       = 0.000001
beta       = 0.5
z_size     = 200
cube_len   = 64
obj_ratio  = 0.5
obj        = 'chair' 

train_sample_directory = '../train_sample/'
model_directory = '../models/'
is_local = False

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def train():

    discriminator = disc_model()
    generator = gen_model()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

    g_optim = Adam(lr=g_lr, beta_1=beta)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")

    d_optim = Adam(lr=d_lr, beta_1=0.9)
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
  
     
    z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
    volumes = d.getAll(obj=obj, train=True, is_local=is_local, obj_ratio=obj_ratio)
    print 'Data loaded .......'
    volumes = volumes[...,np.newaxis].astype(np.float) 

    if not os.path.exists(train_sample_directory):
        os.makedirs(train_sample_directory)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)         

    for epoch in range(n_epochs):
        
        print("Epoch is", epoch)

        idx = np.random.randint(len(volumes), size=batch_size)
        x = volumes[idx]
        z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)

        generated_volumes = generator.predict(z, verbose=0)

        X = np.concatenate((x, generated_volumes))
        Y = np.reshape([1]*batch_size + [0]*batch_size, (-1,1,1,1,1))

        d_loss = discriminator.train_on_batch(X, Y)       
        print("d_loss : %f" % (d_loss))
            
        z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)            
        discriminator.trainable = False
        g_loss = discriminator_on_generator.train_on_batch(z, np.reshape([1]*batch_size, (-1,1,1,1,1)))
        discriminator.trainable = True

        print("g_loss : %f" % (g_loss))

        if epoch % 1000 == 10:
            generator.save_weights(model_directory +'generator_' + str(epoch), True)
            discriminator.save_weights(model_directory +'discriminator_' + str(epoch), True)

        if epoch % 500 == 10:
            generated_volumes = generator.predict(z_sample, verbose=0)
            generated_volumes.dump(train_sample_directory+'/'+str(epoch))            
train()