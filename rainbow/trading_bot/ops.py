"""
Extra Functions
Copyright (c) 2018 Prabhsimran Singh
Licensed under the MIT License (see LICENSE for details)
Written by Prabhsimran Singh
"""

import os
import math
import logging

import numpy as np


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(block):
    """Returns an n-day state representation ending at time t
    """
    #print('get state')
    #d = t - n_days + 1
    #block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]  # pad with t0
    #print(block)
    # full_data_windows=data[t]
    # block=full_data_windows[:,3]
    res = []
    for i in range(len(block)-1):
        # print('blocks')
        # print(block[i])
        # print(block[i + 1] - block[i])
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])
