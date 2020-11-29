import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import constants
from scipy import interpolate


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
            coef[l+i] * np.log2(1 + (np.power(2, 2*i+3) - 1) * t) / (2 * i + 3)
    wave = startpoint * wave
    return wave

def wave_interpolate(wave_old, tf, sample_rate_new):
    t_old = np.linspace(0, tf, len(wave_old))
    f = interpolate.interp1d(t_old, wave_old, kind='quadratic')
    t_step_new = 1. / sample_rate_new
    t_new = np.arange(0, tf, t_step_new)
    wave_new = f(t_new)
    return wave_new


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
            coef[l+i] * np.log2(1 + (np.power(2, 2*i+3) - 1) * t) / (2 * i + 3)
    wave = startpoint * wave

    plt.xlabel("t")
    plt.ylabel("wave")
    plt.plot(t * tf, wave)
    plt.show()

def params_continue_find(params):
    # 限制初始斜率
    a_1 = -1 - np.sum(params)
    coef = np.hstack((a_1, params))
    l = int((len(params) + 1) / 2)
    slope = coef[0]
    for i in range(l):
        slope += (np.power(2, 2*i+3) - 1) / (2 * i + 3) * coef[l+i]
    if slope <= 0 and abs(slope) < 50:
        # 限制上下限
        wave = waveform(1, 0, 1, 3000, params)
        if np.max(wave) <= 1 and np.min(wave) >= 0:
            return False
        else:
            return True
    else:
        return True

def get_random_params_set(min_boundary, max_boundary, params_set_size, startpoint, endpoint, tf, sample_rate):
    rng = np.random.default_rng()
    params_set = np.zeros(shape=(params_set_size, len(min_boundary)))
    
    for i in range(params_set_size):
        params = rng.uniform(min_boundary, max_boundary)
        while params_continue_find(params):
            params = rng.uniform(min_boundary, max_boundary)
        params_set[i] = params
    return params_set


def get_normal_params_set(min_boundary, max_boundary, base_params, std_dev, params_set_size, startpoint, endpoint, tf, sample_rate):
    rng = np.random.default_rng()
    std_dev_scale = std_dev * (np.array(max_boundary) - np.array(min_boundary))
    params_set = np.zeros(shape=(params_set_size, len(min_boundary)))
    for i in range(params_set_size):
        params = rng.normal(base_params, std_dev_scale)
        cond = params >= min_boundary
        params = np.where(cond, params, min_boundary)
        cond = params <= max_boundary
        params = np.where(cond, params, max_boundary)
        while params_continue_find(params):
            params = rng.normal(base_params, std_dev_scale)
            cond = params >= min_boundary
            params = np.where(cond, params, min_boundary)
            cond = params <= max_boundary
            params = np.where(cond, params, max_boundary)
        params_set[i] = params
    return params_set


if __name__ == '__main__':
    # 最速降线
    # params = np.array([-1.15722878, -0.51218646, 1.07800144, -2.35206841, 0.85351334, -0.1725775, -0.012383])   # 0.064683930185784
    # params = np.array([1.32179176, -1.21398091, 0.74667593, -2.54193118, 2.35885283, -1.54051139, 0.43814302])  # 0.038674916857825536
    # params = np.array([-0.70461515, 1.39923573, -0.65270201, -2.51971808, 2.09282611, -1.92157829, 0.68268599])
    # params = np.array([-0.7670372, 1.53994717, -0.71413151, -2.81321045, 2.22119171, -1.8959186, 0.67939656])
    # params = np.array([-1.25691566, 1.64684315, -0.79104391, -3., 1.73756398, -1.89891462, 0.78546396])
    # 数值模拟
    # params = np.array([1.45977357, 0.67831924, -0.58728008, 0.51803271, 2.7090874, -0.60886192, -1.87649613])
    # params = np.array([1.53569274, 1.97344643, -1.31129392, 3., 0.85330547, -3., 0.81725298])   # 0.05954534
    # params = np.array([1.27987736, 2.28957164, -1.38378197, 3.,          1.06099137, -2.97176395, 0.76509959])
    params = np.array([-2.71898154, -0.22467944, 1.00072299, 1.18886061, -2.84190435, 0.36511431, -0.03863528])
    # wave = waveform(1, 0, 1, 50000, params)
    # 随机
    min_boundary = np.array([-3., -3., -3., -4., -4., -4., -4.])
    max_boundary = np.array([3., 3., 3., 4., 4., 4., 4.])
    params = get_random_params_set(min_boundary, max_boundary, 1, 4.5, 0, 3, 20)[0]
    print(params)
    plot_wave(4.5, 0, 3, 50000, params)
    # x = np.linspace(0, 1, 1000)
    # y1 = np.power(x, 1/5)
    # y2 = np.log2((1 + 511 * x)**(1/9))
    # plt.plot(x, y1, label='y1')
    # plt.plot(x, y2, label='y2')
    # plt.legend()
    # plt.show()