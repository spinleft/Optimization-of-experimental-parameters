import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
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
    while True:
        if os.path.exists(filename):
            break
        else:
            time.sleep(1)
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

def waveform_linear(starpoint, endpoint, tf, sample_rate):
    t = np.arange(0, tf, 1 / sample_rate) / tf
    wave = starpoint + (endpoint - starpoint) * t
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

def get_random_params_set(min_boundary, max_boundary, params_set_size):
    rng = np.random.default_rng()
    params_set = np.zeros(shape=(params_set_size, len(min_boundary)))
    
    for i in range(params_set_size):
        params = rng.uniform(min_boundary, max_boundary)
        while params_continue_find(params):
            params = rng.uniform(min_boundary, max_boundary)
        params_set[i] = params
    return params_set


def get_normal_params_set(min_boundary, max_boundary, base_params, std_dev, params_set_size):
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
    # 查看随机波形随机
    min_boundary = np.array([-3., -3., -3., -4., -4., -4., -4.])
    max_boundary = np.array([3., 3., 3., 4., 4., 4., 4.])
    params = get_random_params_set(min_boundary, max_boundary, 1)[0]
    params = np.array([0.43126854, 2.97460144, -1.55703471, 2.74905675, 0.28529145, -1.58702029, 0.27676439])
    print(params)
    plot_wave(1, 0, 1, 1000, params)
