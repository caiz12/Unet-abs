'''
Construct training absorption sample

label the absorption
label 0: no absorption
label 1: absorption
'''

from matplotlib import pyplot as plt
import numpy as np
import pdb

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




xmin=0
xmax=100
arr_len =10000
abs_num = 1000 # No. of absorbers

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
    
# make a plot to double check
ax= plt.subplot(211)
plt.plot(cont.wave, abs.flux, color='red')
ax.set_xlim((10,15))
ax.set_ylim((0.0, 1.3))
# label 0 = no absorption, 1= absorption
ax= plt.subplot(212)
plt.plot(cont.wave, cont.label, color='red')
ax.set_xlim((10,15))
ax.set_ylim((0.0, 1.3))
plt.show()

# make a transformation from 1-D arry to 2-D image
abs_im = np.reshape(cont.flux,(100,100))
lab_im = np.reshape(cont.label,(100,100))
