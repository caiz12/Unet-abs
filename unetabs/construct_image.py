from PIL import Image
from skimage import data, io, filters
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from matplotlib import pyplot as plt
from astropy.io import fits
import pdb

'''
Construct training absorption sample

label the absorption
label 0: no absorption
label 1: absorption

in ipython

   run 
'''


class abs:
    flux = []
    label= [] # 0: no absorber, 1: absorber

class cont:
    wave = []
    flux = []
    label= []
    
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def insert_abs(wave, flux, mu, sig, amp):
    '''
        construct an absorber
        input: wavelength-array
               flux-array
               inserting wavelength: mu
               sigma of the line width: sigma
               amplitude: amp (0 -- 1)
     '''
    abs.flux=  flux - amp*gaussian(wave, mu, sig) #flux
    abs.label= np.zeros(len(wave))
    idx= np.where(abs.flux < 0.99)
    abs.label[idx]= 1.0
    idx= np.where(abs.flux < 0.0)
    abs.flux[idx]= 0.0
    return abs


def continu(xmin, xmax, arr_len):
    '''
        construct 1D continuum
        input:
            wavelength min:  xmin
            wavelength max:  xmax
            number of array: num
    '''
    cont.wave= np.linspace(xmin, xmax, arr_len)
    cont.flux= np.zeros(len(cont.wave))+1
    cont.label=np.zeros(len(cont.wave))
    return cont



def construct_img(xsize, ysize, abs_num): 
    '''
    abs_num:  No. of absorbers
    xsize:  the size of image in x-direction
    ysize:  the size of image in y-direction
    '''
    xmin=0
    xmax= long(xsize* ysize/100.)
    arr_len = long(xsize* ysize) 

# construct continuum
    cont = continu(xmin, xmax, arr_len)
# insert absorber
    abs_pos= np.random.uniform(xmin, xmax, abs_num)
    sig_arr= np.random.normal(0.02, 0.01, abs_num)
    amp_arr= np.random.uniform(0, 1, abs_num)
    for i in range(0, abs_num):
        abs= insert_abs(cont.wave, cont.flux, abs_pos[i], sig_arr[i], amp_arr[i])
        cont.flux= abs.flux
        cont.label= abs.label
    
# make a transformation from 1-D arry to 2-D image
    abs_im = np.reshape(cont.flux,(xsize,ysize))
    lab_im = np.reshape(cont.label,(xsize,ysize))
    abs_lab= np.zeros((xsize, ysize, 2))
    abs_lab[:,:,0]= abs_im
    abs_lab[:,:,1]= lab_im
    return abs_lab

if __name__ == '__main__':

    n=200 #number of images
    for i in range (1,n+1):
        img= construct_img(128, 128, 200)
        abs_arr= img[:,:,0]
        lab_arr= img[:,:,1]
   #Subsection of the image
        print abs_arr.shape
        path=str(i)+'.fits'
        path2=str(i)+'_mask.fits'
        fits.writeto('/Users/zhengcai/Dropbox/data/unet/real/'+path,abs_arr,clobber='True')
        fits.writeto('/Users/zhengcai/Dropbox/data/unet/real/'+path2,lab_arr,clobber='True')
