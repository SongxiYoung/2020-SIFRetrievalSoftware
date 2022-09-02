
# FLD,3FLD,eFLD and SFM
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
import math
from sklearn.decomposition import PCA
import PIL


# all input and output data are array
# Band A is 760nm，band B is 687nm.

def selectRanges(xvals, wlranges):
    # xvals:all range
    # wlranges:selected wavelength range

    idx = np.zeros(xvals.shape)
    if wlranges.ndim == 1:
        for j in range(0, xvals.shape[0]):
            if wlranges[0] <= xvals[j] <= wlranges[1]:
                idx[j] = xvals[j]
    else:
        for i in range(0, wlranges.shape[0]):  # two or more dimensions
            for k in range(0, xvals.shape[0]):  # columns cycle
                if wlranges[i][0] <= xvals[k] <= wlranges[i][1]:
                    idx[k] = xvals[k]
    idx = idx.astype(bool)
    return idx


# select all data in the specific wavelength,then get mean value of them

def get_energy(wavelength, wl):
    index = []
    for i in range(wavelength.shape[0]):
        if 1 > wavelength[i] - wl >= 0:
            index.append(i)
    return np.array(index)

# SVD
# using band ranges to fit Gaussian


def SVD(rad_tra, wavelengths):
    # 1. calculate NDVI
    if wavelengths[0] > 760:
        print("wavelength[0] must be less than 760.")                   # exception handeling
        return 0
    if wavelengths[len(wavelengths)-1] < 760:
        print("wavelength[len(wavelength)-1] must be more than 760.")
        return 0
    wlranges = np.array([760,761])                                      # select window to retrieve SIF
    pixels = selectRanges(wavelengths, wlranges)
    index = np.argwhere(pixels == True)
    R = rad_tra[:, 1]
    NIR = rad_tra[:, 111]
    np.savetxt('R.txt', R, fmt='%f', delimiter=' ', newline='\n')
    np.savetxt('NIR.txt', NIR, fmt='%f', delimiter=' ', newline='\n')
    NDVI = np.array((NIR-R)/(NIR+R))
    np.savetxt('NDVI.txt', NDVI, fmt='%f', delimiter=' ', newline='\n')

    # 2. extract non-vegetation areas
    idx = np.zeros(NDVI.shape)
    count = 0
    for j in range(0, NDVI.shape[0]):
        if -1 <= NDVI[j] <= 0.1:
            idx[j] = NDVI[j]
            count += 1
    idx = idx.astype(bool)
    print(count)

    # 3. calculate non-vegetation principal components
    pca = PCA(n_components = 10)
    sv = pca.fit_transform(np.array(rad_tra[idx, :]).transpose())
    np.savetxt('PCA.txt', sv, fmt='%f', delimiter=' ', newline='\n')

    def func(params, x):
        a1, a2, a3, c1, c2, c3, c4, Fs = params
        v1 = x[:, 0]
        part1 = v1 * (a1*wavelengths + a2*wavelengths*wavelengths + a3*wavelengths*wavelengths*wavelengths)
        part2 = c1*x[:, 0] + c2*x[:, 1] + c3*x[:, 2] + c4*x[:, 3]            # core function
        hf = np.zeros( (1, len(wavelengths)) )
        hf = np.exp(-(wavelengths-740)*(wavelengths-740)/(2*21*21))
        part3 = Fs * hf
        return part1 * part2 + part3

    def error(params, x, y):
        return func(params, x) - y

    # 4. calculate SIF through principal components
    p0 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    SIF = []
    for j in range(rad_tra.shape[0]):
        Para = leastsq(error, p0, args=(sv, rad_tra[j]))                  # leastsq method
        params = Para[0]
        print(j, params[7])
        SIF.append(params[7])
    return np.array(SIF)
    


# 3FLD
# two shoulder band,one bottom band
# equation:Fs=B*[(Lin*Eout-A*Lout*Ein)/(B*Eout-A*Ein)]


def FLD_3_A(radiance, irradiance, wavelength):
    bandout1 = 758.00
    bandin = 760.10
    bandout2 = 770.09

    indexout1 = get_energy(wavelength, bandout1)  # get index

    # print(indexout1)
    wl_out1 = wavelength[indexout1[0]]
    # print(wl_out1)

    indexin = get_energy(wavelength, bandin)
    wl_in = wavelength[indexin[0]]
    # print(wl_in)

    indexout2 = get_energy(wavelength, bandout2)
    wl_out2 = wavelength[indexout2[0]]
    # print(wl_out2)

    # SIF calculation
    Radcalout1 = np.mean(radiance[:, indexout1], axis=1)
    # print(Radcalout1.shape)
    # print(Radcalout1)
    Iradcalout1 = np.mean(irradiance[:, indexout1], axis=1)
    Radcalin = np.mean(radiance[:, indexin], axis=1)
    Iradcalin = np.mean(irradiance[:, indexin], axis=1)
    Radcalout2 = np.mean(radiance[:, indexout2], axis=1)
    Iradcalout2 = np.mean(irradiance[:, indexout2], axis=1)
    w21 = (wl_out2 - wl_in) / (wl_out2 - wl_out1)
    w22 = (wl_in - wl_out1) / (wl_out2 - wl_out1)
    # print(Iradcalin, " ", (w21 * Iradcalout1 + w22 * Iradcalout2))
    e_term = []
    SIF = []
    for i in range(len(Iradcalout2)):
        if Iradcalout2[i] == 0:
            e_term.append(1.1368683772161603e-13)
        else:
            e_term.append(Iradcalin[i] / (w21 * Iradcalout1[i] + w22 * Iradcalout2[i]))
    for j in range(len(Iradcalout2)):
        SIF.append(-(Radcalin[j] - e_term[j] * (w21 * Radcalout1[j] + w22 * Radcalout2[j])) / (1 - e_term[j]))

    return SIF


def FLD_3_B(radiance, irradiance, wavelength):
    bandout1 = 685
    bandin = 687
    bandout2 = 689

    # indexout1=wl_out1.argmin(np.abs(wl_out1))
    indexout1 = get_energy(wavelength, bandout1)
    wl_out1 = wavelength[indexout1[0]]

    indexin = get_energy(wavelength, bandin)
    wl_in = wavelength[indexin[0]]

    indexout2 = get_energy(wavelength, bandout2)
    wl_out2 = wavelength[indexout2[0]]

    Radcalout1 = np.mean(radiance[:, indexout1], axis=1)
    Iradcalout1 = np.mean(irradiance[:, indexout1], axis=1)
    Radcalin = np.mean(radiance[:, indexin], axis=1)
    Iradcalin = np.mean(irradiance[:, indexin], axis=1)
    Radcalout2 = np.mean(radiance[:, indexout2], axis=1)
    Iradcalout2 = np.mean(irradiance[:, indexout2], axis=1)
    w21 = (wl_out2 - wl_in) / (wl_out2 - wl_out1)
    w22 = (wl_in - wl_out1) / (wl_out2 - wl_out1)
    e_term = []
    SIF = []
    for i in range(len(Iradcalout2)):
        if Iradcalout2[i] == 0:
            e_term.append(1.1368683772161603e-13)
        else:
            e_term.append(Iradcalin[i] / (w21 * Iradcalout1[i] + w22 * Iradcalout2[i]))
    for j in range(len(Iradcalout2)):
        SIF.append(-(Radcalin[j] - e_term[j] * (w21 * Radcalout1[j] + w22 * Radcalout2[j])) / (1 - e_term[j]))
    return SIF


# assume that reflectance and fluorescence are constant near absorption band.
# calculation equation: F = (E(out)*L(in)-L(out)*E(in))/(E(out)- E(in)) E:irradiance L:radiance

#FLD
#L=E*r/pi+F


def FLD_A(radiance, irradiance, wavelength):
    bandout = 758.00
    bandin = 760.10

    indexout = get_energy(wavelength, bandout)
    wl_out = wavelength[indexout[0]]

    indexin = get_energy(wavelength, bandin)
    wl_in = wavelength[indexin[0]]

    E_out = np.mean(irradiance[:, indexout], axis=1)
    E_in = np.mean(irradiance[:, indexin], axis=1)
    L_out = np.mean(radiance[:, indexout], axis=1)
    L_in = np.mean(radiance[:, indexin], axis=1)
    SIF = []
    for i in range(len(L_out)):
        #print(L_in[i], " ", L_out[i], " ", E_in[i], " ", E_out[i])
        if L_out[i] == 0:
            SIF.append(1.1368683772161603e-13)
        else:
            #print((E_out[i] * L_in[i] - L_out[i] * E_in[i]), " ", (E_out[i] - E_in[i]))
            SIF.append((E_out[i] * L_in[i] - L_out[i] * E_in[i]) / (E_out[i] - E_in[i]))

    return SIF


def FLD_B(radiance, irradiance, wavelength):
    bandout = 686
    bandin = 687

    indexout = get_energy(wavelength, bandout)
    wl_out = wavelength[indexout[0]]

    indexin = get_energy(wavelength, bandin)
    wl_in = wavelength[indexin[0]]

    E_out = np.mean(irradiance[:, indexout], axis=1)
    E_in = np.mean(irradiance[:, indexin], axis=1)
    L_out = np.mean(radiance[:, indexout], axis=1)
    L_in = np.mean(radiance[:, indexin], axis=1)
    SIF = []
    for i in range(len(L_out)):
        # print(L_in[i], " ", L_out[i], " ", E_in[i], " ", E_out[i])
        if L_out[i] == 0:
            SIF.append(1.1368683772161603e-13)
        else:
            # print((E_out[i] * L_in[i] - L_out[i] * E_in[i]), " ", (E_out[i] - E_in[i]))
            SIF.append((E_out[i] * L_in[i] - L_out[i] * E_in[i]) / (E_out[i] - E_in[i]))
    return SIF


def SFM_A(radiance, irradiance, wavelengths, flag):
    wlranges = np.array([759, 767.6])  # select window to retrieve SIF
    pixels = selectRanges(wavelengths, wlranges)

    wavelengths = wavelengths[pixels]
    radiance = radiance[:, pixels]
    irradiance = irradiance[:, pixels]
    if flag == "constant":
        def func(params, x):

            a, b = params
            return x * a + b

        def error(params, x, y):
            return func(params, x) - y

        p0 = np.array([1, 1])  # the initial value of k,b
        SIF = []
        for j in range(irradiance.shape[0]):
            # print(irradiance[j])
            # print(radiance[j])
            Para = leastsq(error, p0, args=(irradiance[j], radiance[j]))
            params = Para[0]
            if params[1] == 1.1368683772161603e-13:
                SIF.append(params[1])
            else:
                SIF.append(-params[1])
        return np.array(SIF)

    if flag == "linear":
        def func(params, x):

            a, b, c, d = params
            return x * (a + 760 * b) + (c + 760 * d)

        def error(params, x, y):
            return func(params, x) - y

        p0 = np.array([1, 1, 1, 1])
        SIF = []
        for j in range(irradiance.shape[0]):
            Para = leastsq(error, p0, args=(irradiance[j], radiance[j]))
            params = Para[0]
            # print(params[2], " ", params[3], " ", params[2] + params[3] * 760)
            SIF.append(-(params[2] + params[3] * 760))
        return np.array(SIF)

    if flag == "quar":
        def func(params, x):

            a, b, c, d, e = params
            return x * (a + 760 * b) + (c + 760 * d + 760 * 760 * e)

        def error(params, x, y):
            return func(params, x) - y

        p0 = np.array([1, 1, 1, 1, 1])
        SIF = []
        for j in range(irradiance.shape[0]):
            Para = leastsq(error, p0, args=(irradiance[j], radiance[j]))
            params = Para[0]
            SIF.append(params[2] + params[3] * 760 + params[4] * 760 * 760)
        return np.array(SIF)


def SFM_B(radiance, irradiance, wavelengths, flag):
    wlranges = np.array([686, 691])

    pixels = selectRanges(wavelengths, wlranges)
    wavelengths = wavelengths[pixels]
    radiance = radiance[:, pixels]
    irradiance = irradiance[:, pixels]

    if flag == "constant":
        def func(params, x):

            a, b = params
            return x * a + b

        def error(params, x, y):
            return func(params, x) - y

        p0 = np.array([1, 1])
        SIF = []
        for j in range(irradiance.shape[0]):
            Para = leastsq(error, p0, args=(irradiance[j], radiance[j]))
            params = Para[0]
            SIF.append(params[1])
        return np.array(SIF)

    if flag == "linear":
        def func(params, x):

            a, b, c, d = params
            return x * (a + 687 * b) + (c + 687 * d)

        def error(params, x, y):
            return func(params, x) - y

        p0 = np.array([1, 1, 1, 1])
        SIF = []
        for j in range(irradiance.shape[0]):
            Para = leastsq(error, p0, args=(irradiance[j], radiance[j]))
            params = Para[0]
            SIF.append(params[2] + params[3] * 687)
        return np.array(SIF)

    if flag == "quar":
        def func(params, x):

            a, b, c, d, e = params
            return x * (a + 687 * b) + (c + 687 * d + 687 * 687 * e)

        def error(params, x, y):
            return func(params, x) - y

        p0 = np.array([1, 1, 1, 1, 1])
        SIF = []
        for j in range(irradiance.shape[0]):
            Para = leastsq(error, p0, args=(irradiance[j], radiance[j]))
            params = Para[0]
            SIF.append(params[2] + params[3] * 687 + params[4] * 687 * 687)
        return np.array(SIF)

