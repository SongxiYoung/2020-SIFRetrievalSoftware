import numpy as np
from scipy import misc
import pandas as pd
import SIFretrieval
import matplotlib.pyplot as plt
import os
import pylab as pl
import PIL
import math


def simulate_irra(wv, p_irra, wavelength):  # function to simulate irradiances
    """
    :param wv:  known wavelength array
    :param p_irra:  point irradiances array
    :param wavelength:  total wavelength array
    :return:  irradiance
    """
    n = len(wv)
    T_ave = []
    irradiances = []
    c = 2.997925e+8
    h = 6.626e-34
    k = 1.38054e-23
    for ii in range(n):
        part1 = 2*np.pi*h*c**2 / (p_irra[ii]/1000 * (wv[ii]*math.pow(10, -9))**5)
        T_ave.append(h*c / (wv[ii]*math.pow(10, -9)*k*math.log( part1 + 1, math.e) ))
    T = np.mean(T_ave) * 0.47
    print("T: ", T)
    for jj in range(len(wavelength)):
        irradiances.append(2*np.pi*h*c**2*1000 / (wavelength[jj]*math.pow(10, -9))**5 / (np.exp(h*c / (wavelength[jj]*math.pow(10, -9)*k*T))-1) )

    '''give plot of diurnal SIF'''
    fig = plt.figure(dpi=300)
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.12, right=0.9, top=0.88, bottom=0.2, hspace=0.08, wspace=0.1)
    ax.plot(wavelength, irradiances)
    ax.set_xlabel("wavelength", fontsize=8)
    ax.set_ylabel("irradiance(mW/m2/nm/sr)", fontsize=8)
    plt.title("Irradiance Simulation",fontsize = 12)
    pl.xticks(rotation=360)
    plt.savefig('irradiance.png')
    plt.close()

    return irradiances


#-------------------------------1. input the hyperspectral remote sensing files--------------------------------------------
filedir = r"example data"  # the directory of files(.csv)
savedir = r"result"        # the path of result saved
name_list = os.listdir(filedir)
for item in name_list:
        if item.startswith('.') and os.path.isfile(os.path.join(filedir, item)):
            name_list.remove(item)

wavelength = np.loadtxt(r"wavelength.txt")
print("wavelength:",len(wavelength))

for lll in range(len(name_list)):
    fullpath = os.path.join(filedir, name_list[lll])  # get the fullpath of file
    savepath = os.path.join(savedir, name_list[lll])  # get the savepath of result
    result_SIF = pd.DataFrame()
    print("inputfile path:", fullpath)
    print("savefile path:", savepath)
    data = np.loadtxt(fullpath, dtype=float, skiprows=0, comments=';')   #1190*530*127
    print("input data size:", data.shape)

    columns = np.arange(data.shape[1])
    rows = np.arange(data.shape[0])
    data = data/10000     #units conversion  mW/m2/nm/sr

    o_radiance = np.zeros(data.shape)

    #--------------------2. convert format:bil to bsq---------------------------------------
    raw_row = int(data.shape[0]/len(wavelength))
    for j in range(0, raw_row):  # 1190
        for i in range(0, len(wavelength)):  # 127
            o_radiance[i * raw_row + j, :] = data[j * len(wavelength) + i, :]

    radiance = np.zeros((int(data.shape[0]*data.shape[1]/len(wavelength)), len(wavelength)))

    for i in range(0, len(wavelength)):
        temp = np.array(o_radiance[(i*raw_row):(i+1)*raw_row-1, :]).reshape(-1, 1)
        for j in range(0, temp.shape[0]):
            radiance[j, i] = temp[j]

    print("reshaped data size:", radiance.shape)

    #--------------------3. simulate irradiances---------------------------------------

    wv = [305, 310, 324, 380]
    p_irra = [67.0538, 100.3241, 337.6124, 603.4009]
    irra_data = simulate_irra(wv, p_irra, wavelength)
    irra_data = np.array(irra_data).astype(np.float)
    irradiance = np.zeros(radiance.shape)
    for i in range(0, raw_row):
        for j in range(0, o_radiance.shape[1]):
            if o_radiance[i, j] == 0:
                irradiance[i*o_radiance.shape[1]+j, :] = 0;
            else:
                irradiance[i*o_radiance.shape[1]+j, :] = irra_data[:];
    irradiance = irradiance.astype(np.float)
    np.savetxt('irradiance.txt', irradiance, fmt='%f', delimiter=' ', newline='\n')

    #--------------------------4. calculate SIF using SIFretrieval.py-----------------------
    SIF_SVD = SIFretrieval.SVD(radiance, wavelength)
    result_SIF = pd.DataFrame({"SVD": SIF_SVD})             #save into DataFrame

    #--------------------------5. show SIF pictures-----------------------
    print(result_SIF["SVD"].values.shape)
    image_array = np.array(result_SIF["SVD"].values.reshape(raw_row, data.shape[1])) #convert to plt
    plt.imshow(image_array)
    plt.axis('off')
    plt.show()
    im = PIL.Image.fromarray(image_array.astype(np.float))
    im.save('Image_result_SVD.tif') 

print("finish!!!")
