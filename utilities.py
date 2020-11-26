import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import constants


def get_dict_from_file(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, 'r') as in_file:
        tdict_string = ''
        for line in in_file:
            temp = (line.partition('#')[0]).strip('\n').strip()
            if temp != '':
                tdict_string += temp + ','
        in_file.close()
    array = np.array
    inf = float('inf')
    nan = float('nan')
    tdict = eval('dict(' + tdict_string + ')')
    return tdict


def save_params_to_file(filename, params):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, 'w') as out_file:
        for param in params:
            out_file.write('%.5f\n' % param)
        out_file.close()


def get_result_from_file(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    result = []
    with open(filename, 'r') as in_file:
        for line in in_file:
            temp = line.strip('\n')
            if temp != '':
                result.append(float(temp))
        in_file.close()
    result = np.array(result)
    return result


def get_datetime_now_string():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')


def save_dict_to_txt_file(tdict, filename):
    with open(filename, 'w') as out_file:
        for key in tdict:
            out_file.write(
                str(key) + '=' + repr(tdict[key]).replace('\n', '').replace('\r', '') + '\n')
        out_file.close()


def waveform(startpoint, endpoint, tf, sample_rate, params):
    num_params = len(params)
    a_1 = endpoint / startpoint - 1 - np.sum(params)
    coef = np.hstack((a_1, params))
    l = int((num_params + 1) / 2)
    t_step = 1.0 / sample_rate
    t = np.array(np.arange(0, tf, t_step)) / tf
    wave = np.ones(len(t))
    for i in range(l):
        wave = wave + coef[i] * np.power(t, i+1) + \
            coef[l+i] * np.power(t, 1/float(i+2))
    wave = startpoint * wave
    return wave


def plot_wave(startpoint, endpoint, tf, sample_rate, params):
    num_params = len(params)
    a_1 = endpoint / startpoint - 1 - np.sum(params)
    coef = np.hstack((a_1, params))
    l = int((num_params + 1) / 2)
    t_step = 1.0 / sample_rate
    t = np.array(np.arange(0, tf, t_step)) / tf
    wave = np.ones(len(t))
    for i in range(l):
        wave = wave + coef[i] * np.power(t, i+1) + \
            coef[l+i] * np.power(t, 1/float(i+2))
    wave = startpoint * wave

    plt.xlabel("t")
    plt.ylabel("wave")
    plt.plot(t * tf, wave)
    plt.show()

def params_continue_find(min_boundary, max_boundary, startpoint, endpoint, sample_rate, params):
    K_0 = 12 * constants.Boltzmann * 1.5e-6
    wave = waveform(startpoint, endpoint, params[-1], sample_rate, params[:-1])
    wave_diff = np.diff(wave)
    if np.max(wave) <= startpoint and np.min(wave) >= endpoint and np.max(np.abs(wave_diff)) < (startpoint - endpoint) / 20:
        return False
    else:
        return True

def get_random_params_set(min_boundary, max_boundary, params_set_size, startpoint, endpoint, sample_rate):
    rng = np.random.default_rng()
    params_set = np.zeros(shape=(params_set_size, len(min_boundary)))
    
    for i in range(params_set_size):
        params = rng.uniform(min_boundary, max_boundary)
        while params_continue_find(min_boundary, max_boundary, startpoint, endpoint, sample_rate, params):
            params = rng.uniform(min_boundary, max_boundary)
        params_set[i] = params
    return params_set


def get_normal_params_set(min_boundary, max_boundary, base_params, std_dev, params_set_size, startpoint, endpoint, sample_rate):
    rng = np.random.default_rng()
    std_dev_scale = std_dev * (np.array(max_boundary) - np.array(min_boundary))
    params_set = np.zeros(shape=(params_set_size, len(min_boundary)))
    for i in range(params_set_size):
        params = rng.normal(base_params, std_dev_scale)
        cond = params >= min_boundary
        params = np.where(cond, params, min_boundary)
        cond = params <= max_boundary
        params = np.where(cond, params, max_boundary)
        while params_continue_find(min_boundary, max_boundary, startpoint, endpoint, sample_rate, params):
            params = rng.normal(base_params, std_dev_scale)
            cond = params >= min_boundary
            params = np.where(cond, params, min_boundary)
            cond = params <= max_boundary
            params = np.where(cond, params, max_boundary)
        params_set[i] = params
    return params_set


if __name__ == '__main__':
    # params = np.array([-2.71779894, 2.65982562, -0.84004817, -2.42029854, -0.011407, 0.64039272, -0.28129203])
    # params = np.array([-1.15722878, -0.51218646, 1.07800144, -2.35206841, 0.85351334, -0.1725775, -0.012383])   # 0.064683930185784
    # params = np.array([1.32179176, -1.21398091, 0.74667593, -2.54193118, 2.35885283, -1.54051139, 0.43814302])  # 0.038674916857825536
    # params = np.array([0.62040317, 1.1948361, -0.59836405, -0.10365333, -0.79325315, 2.10052768, -1.23012641, 4.14534571])
    params = np.array([1.94977766, 2.60590206, -2.7299242, -1.60359176, -1.16171635, 2.38431112, -0.98774287, 3.16765493])
    # params = np.array([-0.5900328, 2.77703416, -1.31402892, 0.19818796, 0.64368691, -2.09109619, 1.1801839, 5.])
    plot_wave(4.2644e-28, 4.2644e-28 / 25, params[-1], 200, params[:-1])
