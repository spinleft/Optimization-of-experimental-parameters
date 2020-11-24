import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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

    result = np.array([])
    with open(filename, 'r') as in_file:
        for line in in_file:
            temp = line.strip('\n')
            if temp != '':
                result = np.hstack((result, float(temp)))
        in_file.close()
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
    plt.plot(t, wave)
    plt.show()


def get_random_params_set(min_boundary, max_boundary, params_set_size, startpoint, endpoint, sample_rate):
    rng = np.random.default_rng()
    for i in range(params_set_size):
        flag = True
        while flag:
            params = rng.uniform(min_boundary, max_boundary)
            wave = waveform(startpoint, endpoint, params[-1], sample_rate, params[:-1])
            wave_diff = np.diff(wave)
            if np.max(wave) < 4.2644e-28 and np.min(wave) > 4.2644e-28 / 50 and np.max(np.abs(wave_diff)) < 4.2644e-28 / 50:
                params_set = np.array([params]) if (i == 0) else np.vstack((params_set, params))
                flag = False
    return params_set

def get_normal_params_set(min_boundary, max_boundary, base_params, std_dev, params_set_size, startpoint, endpoint, sample_rate):
    rng = np.random.default_rng()
    std_dev_scale = std_dev * (np.array(max_boundary) - np.array(min_boundary))
    for i in range(params_set_size):
        flag = True
        while flag:
            params = rng.normal(base_params, std_dev_scale)
            cond = params >= min_boundary
            params = np.where(cond, params, min_boundary)
            cond = params <= max_boundary
            params = np.where(cond, params, max_boundary)
            wave = waveform(startpoint, endpoint, params[-1], sample_rate, params[:-1])
            wave_diff = np.diff(wave)
            if np.max(wave) < 4.2644e-28 and np.min(wave) > 4.2644e-28 / 50 and np.max(np.abs(wave_diff)) < 4.2644e-28 / 50:
                params_set = np.array([params]) if (i == 0) else np.vstack((params_set, params))
                flag = False
    return params_set

def get_remotest_params(min_boundary, max_boundary, train_params_set):
    num_params = len(min_boundary)
    # 对参数集合在每个维度上独立排序
    params_sort = np.sort(train_params_set, axis=0)
    # 补充上下限，方便索引
    params_sort = np.vstack((min_boundary, params_sort))
    params_sort = np.vstack((params_sort, max_boundary))
    # 求每个维度的参数最大间距
    params_diff = np.diff(params_sort, n=1, axis=0)
    max_diff_indexs = np.argmax(params_diff, axis=0)
    # 做一点扰动
    cond = np.random.rand() < 0.8
    max_diff_indexs = np.where(
        cond, max_diff_indexs, np.random.randint(0, len(params_diff)))
    # 计算最大间隔中的参数
    low_indexs = (tuple(max_diff_indexs), tuple(i for i in range(num_params)))
    high_indexs = (tuple(max_diff_indexs + 1),
                   tuple(i for i in range(num_params)))
    remotest_params = (params_sort[high_indexs] - params_sort[low_indexs]) / 2

    return remotest_params


if __name__ == '__main__':
    # params = np.array([-2.71779894, 2.65982562, -0.84004817, -2.42029854, -0.011407, 0.64039272, -0.28129203])
    # params = np.array([-1.15722878, -0.51218646, 1.07800144, -2.35206841, 0.85351334, -0.1725775, -0.012383])   # 0.064683930185784
    params = np.array([1.32179176, -1.21398091, 0.74667593, -2.54193118, 2.35885283, -1.54051139, 0.43814302])  # 0.038674916857825536
    plot_wave(10, 0, 15.71, 5000, params)
