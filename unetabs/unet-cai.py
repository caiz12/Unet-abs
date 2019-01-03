####################################################3
# CHANGE LOG 
# 5-24-2018:  Retraining model on improved galsim simulation set, with noise variation and galaxy center offset.
# 7-13-2018:  Retraining model on new galsim simulation set with fix in random centering offset and galaxy, clump property catalog generation.
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from matplotlib import pyplot as plt

from keras.callbacks import EarlyStopping, ModelCheckpoint
from glob import glob
import numpy as np
from astropy.io import fits
import random
import sys, os
import pdb

__all__ = ['UNet']


# Read input data.
# Images are of size (img_size, img_size), stored in FITS format in directory "input_path".
def read_files (input_path, img_size):

    # first, lets read in all the fits files in the relevant directory
    files = glob(input_path+'/train_img/*.fits')
    nn = len(files)

    # allocate
    img_train = np.empty((nn,img_size,img_size))
    img_mask_train = np.empty((nn,img_size,img_size))

    print ("Reading in "+str(nn)+" fits files...")

    for i in range (0,nn):

        try:
            hdu = fits.open(files[i])
            img_train[i] = hdu[0].data
            img_mask_train[i] = hdu[1].data
        except Exception as excep:
            print (excep)
    # return images, masks, and vector of corresponding galaxy ids
    return img_train, img_mask_train


# Read CANDELS images
def read_files_real (input_path, img_size, nn = None):

    # first, lets read in all the fits files in the relevant directory
    files = glob(input_path+'/*.fits')

    if (nn is None):
        nn = len(files)

    # allocate
    imgs = np.empty((nn,img_size,img_size))

    print ("Reading in "+str(nn)+" fits files...")

    dot_freq = int(np.round(nn/100.+0.5))

    for i in range (0,nn):
        if (i % dot_freq == 0):
            sys.stdout.write(str(i/dot_freq)+".")
            sys.stdout.flush()
        try:
            hdu = fits.open(files[i])
            imgs[i] = hdu[0].data
        except Exception as excep:
            print (excep)

    return imgs


def UNet(input_shape, init_filt_size=64):
    """
    U-Net image segmentation model

    Parameters
    ----------
    input_shape : tuple of ints
        Shape of the input images
        The order depends on which keras backend is used.
            - TensorFlow => (row_size, col_size, channels)
            - Theano => (channels, row_size, col_size)
    init_filt_size : int, optional
        Size of the first filter (default is 64)
        It determines automatically the size of the next filters

    Returns
    -------
    model : Keras model

    References
    ----------
    https://arxiv.org/abs/1505.04597v1

    """
    img_input = Input(shape=input_shape)

    nfilt1 = init_filt_size
    nfilt2 = nfilt1 * 2
    nfilt3 = nfilt2 * 2

    # Block 1
    x = Conv2D(nfilt1, (3, 3), activation='relu', padding='same',
               name='block1_conv1')(img_input)
    x_1a = Conv2D(nfilt1, (3, 3), activation='relu', padding='same',
                  name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),
                     name='block1_pool')(x_1a)

    # Block 2
    x = Conv2D(nfilt2, (3, 3), activation='relu', padding='same',
               name='block2_conv1')(x)
    x_2a = Conv2D(nfilt2, (3, 3), activation='relu', padding='same',
                  name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),
                     name='block2_pool')(x_2a)

    # Block 3
    x = Conv2D(nfilt3, (3, 3), activation='relu', padding='same',
               name='block3_conv1')(x)
    x = Conv2D(nfilt3, (3, 3), activation='relu', padding='same',
               name='block3_conv2')(x)
    x_2b = Conv2DTranspose(nfilt2, (2, 2), strides=(2, 2),
                           input_shape=(None, 23, 23, 1),
                           name='block3_deconv1')(x)

    # Deconv Block 1
    x = concatenate([x_2a, x_2b])
    x = Conv2D(nfilt2, kernel_size=(3, 3), activation='relu',
               padding='same', name='dblock1_conv1')(x)
    x = Conv2D(nfilt2, (3, 3), activation='relu', padding='same',
               name='dblock1_conv2')(x)
    x_1b = Conv2DTranspose(nfilt1, kernel_size=(2, 2), strides=(2, 2),
                           name='dblock1_deconv')(x)

    # Deconv Block 2
    x = concatenate([x_1a, x_1b], input_shape=(None, 92, 92, None),
                    name='dbock2_concat')
    x = Conv2D(nfilt1, (3, 3), activation='relu', padding='same',
               name='dblock2_conv1')(x)
    x = Conv2D(nfilt1, (3, 3), activation='relu', padding='same',
               name='dblock2_conv2')(x)

    # NOTE: this line hardcodes the number of output channels (currently == 1).
    # Output convolution.
    x = Conv2D(1, (1, 1), activation=None, padding='same',
               name='dblock2_conv3')(x)

    # Create model
    model = Model(img_input, x, name='UNet')

    return model





'''
 main program starts here
 run unet-cai.py
 
'''
if __name__ == '__main__':
    
    batch_size = 4
    epochs = 30
    verbose = 1
    img_size = 128
    max_n = 100           # size of training set (remainder to be used for test set)
    ntrain = max_n*9/10     # training / validation split
    nval = max_n/10

    date = '_09_09_18'      # date suffix to keep track of models

    dir = '/Users/zhengcai/Dropbox/data/unet'  #set directory for the input images 
    img_dir = dir 
    saved_outputs = dir+'/saved_outputs'       #set directory for the output images 

    model_name = dir+'/models/unet_'+str(max_n)+date  # model date 

    model = UNet ((img_size,img_size,1))
    model.compile(optimizer = SGD(lr = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    READ_DATA = 0
    TRAIN = 0
    PREDICT = 0
    VERIFY= 1
    PREDICT_REAL = 0
    VERIFY_REAL = 0

    if READ_DATA:
        print ("Reading input data...")
    
        imgs, img_masks = read_files (img_dir, img_size)
    
        #save above input imgs and img_masks to .npy files 
        np.save(saved_outputs+'/imgs_'+date+'.npy', imgs)
        np.save(saved_outputs+'/img_masks_'+date+'.npy', img_masks)
    
        '''
        if input imgs and img_masks already in .npy format, then
        directly read imgs and img_masks using the following two
        commands.
    
        print ("Loading images..")
        imgs = np.load(saved_outputs+'/imgs_'+date+'.npy')
    
        print ("Loading masks...")
        img_masks = np.load(saved_outputs+'/img_masks_'+date+'.npy')
        '''
        print ("done.")
        imgs = np.expand_dims(imgs,3)
        img_masks = np.expand_dims(img_masks,3)

    if (TRAIN):
    
        print ("Preparing data...")
    
        ind=random.sample(range(0, max_n), ntrain+nval-1)
        img_train = imgs[ind[0:ntrain],:,:,:]   
        img_val = imgs[ind[ntrain:ntrain+nval],:,:,:]
        img_mask_train = img_masks[ind[0:ntrain],:,:,:]
        img_mask_val = img_masks[ind[ntrain:ntrain+nval],:,:,:]
    
        print ("done.")
    
        # use previous best model if desired
        if (os.path.isfile(model_name+"_best.hd5")):
            model.load_weights(model_name+"_best.hd5")

        model_checkpoint =  ModelCheckpoint (model_name+"_best.hd5",\
                                            monitor='val_loss', \
                                         verbose=verbose,save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        print ("Fitting model...")
        history = model.fit(img_train, img_mask_train, batch_size=batch_size, epochs=epochs,\
                            validation_data=(img_val, img_mask_val), shuffle=True, \
                            verbose=verbose, callbacks=[model_checkpoint, early_stopping])
        
        print ("done")


    if (PREDICT):

        print ("Checking model predictions...")
    
        model.load_weights(model_name+"_best.hd5")
        img_test = imgs[max_n:]
        img_mask_test = img_masks[max_n:]
        img_mask_pred = model.predict(img_test)
    
        (n, m , _, _) = img_test.shape 
    
        test_output = np.zeros((n,m,m,3))
        test_output[:,:,:,0] = img_test[:,:,:,0]
        test_output[:,:,:,1] = img_mask_test[:,:,:,0]
        test_output[:,:,:,2] = img_mask_pred[:,:,:,0]
    
        print ("done.")
    
        print ("Saving test results...")
    
        np.save(saved_outputs+'/img_test-'+str(img_test.shape[0])+date+'.npy', test_output)



    if (VERIFY):

        imgs_pred = np.load(saved_outputs+'/img_test-100_09_09_18.npy')
    
        for i in range(len(imgs_pred[:,0,0,0])):
            imgs_i_img= imgs_pred[i,:,:,0]
            imgs_i_msk= imgs_pred[i,:,:,1]
            imgs_i_pre= imgs_pred[i,:,:,2]

            path=str(i)+'.fits'
            path2=str(i)+'_test_mask.fits'
            path3=str(i)+'_pred_mask.fits'

            fits.writeto('/Users/zhengcai/Dropbox/data/unet/veri0_fits/'+path,imgs_i_img,clobber='True')
            fits.writeto('/Users/zhengcai/Dropbox/data/unet/veri0_fits/'+path2,imgs_i_msk,clobber='True')
            fits.writeto('/Users/zhengcai/Dropbox/data/unet/veri0_fits/'+path3,imgs_i_pre,clobber='True')

    
    if (PREDICT_REAL):
    
        print ("Reading real galaxy images...")
   
    # choose the filter we are using (based on the filter the model was trained with)

    # read real image
        print ("Loading images..")
        img_dir = img_dir + '/real'
        imgs_real = read_files_real (img_dir, img_size, nn = None)
        np.save(saved_outputs+'/imgs_real_'+ date +'.npy', imgs_real)
    
    #imgs_real = np.load(saved_outputs+'/imgs_real_'+ date +'.npy')
    
        print ("done.")

        imgs_real = np.expand_dims(imgs_real,3)

        print ("Checking model predictions for real absorption...")
    
        model.load_weights(model_name+"_best.hd5")
        imgs_real_mask_pred = model.predict(imgs_real)
    
        (nn, m , _, _) = imgs_real.shape 
    
        test_output = np.zeros((nn,m,m,2))
        test_output[:,:,:,0] = imgs_real[:,:,:,0]
        test_output[:,:,:,1] = imgs_real_mask_pred[:,:,:,0]
    
        print ("done.")
    
        print ("Saving test results...")
    
        np.save(saved_outputs+'/imgs_real_pred_'+date+'.npy', test_output)


    if (VERIFY_REAL):

        imgs_pred = np.load(saved_outputs+'/imgs_real_pred__09_05_18.npy')

        for i in range(len(imgs_pred[:,0,0,0])):
            imgs_i_img= imgs_pred[i,:,:,0]
            imgs_i_msk= imgs_pred[i,:,:,1]
            path=str(i)+'.fits'
            path2=str(i)+'_pred_mask.fits'
            fits.writeto('/Users/zhengcai/Dropbox/data/unet/veri_fits/'+\
                        path,imgs_i_img,clobber='True')
            fits.writeto('/Users/zhengcai/Dropbox/data/unet/veri_fits/'+\
                        path2,imgs_i_msk,clobber='True')

