#!/usr/bin/env python
# coding: utf-8

import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy.stats import bernoulli


class DataGenerator(keras.utils.Sequence):
    """Generates Data for Keras"""
    
    def __init__(self, list_IDs, frames_dir, masks_dir, batch_size=32, dim=(352,512,35),n_channels=2, n_classes = 3, shuffle=True):
        
        """Initialize Generator"""
        
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        
    
    def augmentation_params(self, shift_range=0, rotate_range=0, zoom_range=0, augment=False, normalize=False, aug_prob = 0.5):
        """--shift_range = a, a is fraction between (0,1): causes shift in x between (-a*dim_x,a*dim_x), similarly in y
           --rotate_range = a, a is in degrees: causes rotation in degrees between (-rotate_range, rotate_range)
           --zoom = a, a is a fraction between (0,1): causes zoom between (1-a) to (1+a)
           --augment = a, a is binary True or False: setting false means no augmentation
           --normalize = a, a is binary True or False: setting false means no normalization(turned off for masks automatically)"""
        
        self.shift = shift_range
        self.rotate = rotate_range
        self.zoom = zoom_range
        self.augment = augment
        self.normalize = normalize
        self.aug_prob = aug_prob
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs)/self.batch_size))
    
    def __getitem__(self, index):
        
        'Generate batches of data pertaining to supplied batch index'
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=int)
        
        if(self.n_channels == 1):
            
            for i, ID in enumerate(list_IDs_temp):
            # Store sample

                seed = np.random.randint(10000, size=1)[0]
                temp_x = np.load(self.frames_dir + '/frame_' + str(ID) + '.npy')
                temp_x = augmentation(temp_x, self.shift, self.rotate, self.zoom,self.augment, self.normalize, self.aug_prob, seed)
                X[i, :, :, :, 0] = temp_x

                # Store class
                
                temp_y = np.load(self.masks_dir + '/mask_' + str(ID) + '.npy')
                temp_y = augmentation(temp_y, self.shift, self.rotate, self.zoom,self.augment, False, self.aug_prob, seed)
                y[i, :, :, :, 0] = temp_y

        else:
            
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample - Normalizing(Try without it also)
                seed = np.random.randint(10000, size=1)[0]

                temp_x = np.load(self.frames_dir + '/frame_' + str(ID) + '.npy')
                temp_x = augmentation(temp_x, self.shift, self.rotate, self.zoom, self.augment, self.normalize, self.aug_prob, seed)
                X[i] = temp_x
                
                

                # Store class
                temp_y = np.load(self.masks_dir + '/mask_' + str(ID) + '.npy')
                temp_y = augmentation(temp_y, self.shift, self.rotate, self.zoom, self.augment, False, self.aug_prob, seed)
                y[i] = temp_y

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    
def augmentation(x, shift_range, rotate_range, zoom_range, augment=False,normalize=False, aug_prob = 0.5, seed = 1):
    
    """FEED IMAGE AS (X,Y,Z) 
       AUGMENTATION IN (X,Y) DIMS"""
    
    x_transformed = x
    
    if(augment == True):
        np.random.seed(seed)
        r = bernoulli.rvs(aug_prob, size=1)
        
        if(r[0] == 1):

          image1_datagen = ImageDataGenerator()

          shift = np.linspace(-1*shift_range, 1*shift_range, 101)
          
          index_x = np.random.choice(shift.shape[0], 1, replace=True)
          shift_x = int(shift[index_x] * x.shape[0])
          index_y = np.random.choice(shift.shape[0], 1, replace=True)
          shift_y = int(shift[index_y] * x.shape[1])
          
          rotate = np.linspace(-1*rotate_range, 1*rotate_range, 2*rotate_range+1)
          
          index_r = np.random.choice(rotate.shape[0], 1, replace=True)
          deg = int(rotate[index_r])
          
          zoom = np.linspace(1-zoom_range, 1+zoom_range, 101)
          
          index_zx = np.random.choice(zoom.shape[0], 1, replace=True)
          shift_zx = zoom[index_x][0]

          index_zy = np.random.choice(zoom.shape[0], 1, replace=True)
          shift_zy = zoom[index_y][0]
          
          data_gen_args = dict(theta=deg,
                            tx=shift_x,
                            ty=shift_y,
                            zx=shift_zx,
                            zy=shift_zy)
          
          #print(data_gen_args)
          
          x_transformed = image1_datagen.apply_transform(x, transform_parameters=data_gen_args)
    
    if(normalize == True):
        min_val = np.min(x_transformed)
        max_val = np.max(x_transformed)
        if(max_val != min_val):
            x_transformed = (x_transformed - min_val)/(max_val - min_val+0.00001)
        
    return x_transformed
    

# In[ ]:



