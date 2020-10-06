# 离散小波变换

import pywt
import numpy as np


def getDWT(signals, name, num, mode):
    '''
    :param signals: the original signal
    :param name: the wavlet name to use
    :param mode: the way of the decomposition
    :param num: the decomposition level
    :return: cD:  a N-row matrix containing the detail coefficients up to N levels
             cA:  the same for the approximations
    '''

    # perform CWT for Morlet

    if name == "Morlet":
        c = pywt.cwt(signals, name, num)

        cD = c
        cA = c
    else:
        # perform wavlet decomposition

        coeff = pywt.wavedec(signals, wavelet=name, mode=mode, level=num)

        cA = coeff[0]
        cD = coeff[1:]

    return cA, cD
