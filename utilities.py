import os
import time
import h5py
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


def waveform_polyn(startpoint, endpoint, tf, sample_rate, params):
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


def waveform_interpolate(startpoint, endpoint, tf, sample_rate, params):
    if len(params) % 2 == 1:
        print("num_params must be an even number")
        raise ValueError
    num_samples = len(params) // 2
    t_scale = (1. - params[:num_samples]).cumprod()
    t_samples = (np.concatenate(([1], t_scale[:-1])) - t_scale).cumsum() * tf
    t_samples = np.concatenate(([0.], t_samples, [tf]))
    wave_samples = params[num_samples:].cumprod(
    ) * (startpoint - endpoint) + endpoint
    wave_samples = np.concatenate(([startpoint], wave_samples, [endpoint]))
    return cubic_interpolate(t_samples, wave_samples, sample_rate)


def cubic_interpolate(t_samples, wave_samples, sample_rate):
    f = interpolate.interp1d(t_samples, wave_samples, kind='cubic')
    t_step = 1. / sample_rate
    t_interp = np.arange(0, t_samples[-1], t_step)
    wave_interp = f(t_interp)
    return wave_interp


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

def print_archive(archive_filename):
    f = h5py.File(archive_filename, 'r')
    print("****** 实验参数 ******")
    num_params = f['num_params'][()]
    print("---- num_params = %d ----" % num_params)
    min_boundary = f['min_boundary'][()]
    print("---- min_boundary = " + repr(min_boundary) + " ----")
    max_boundary = f['max_boundary'][()]
    print("---- max_boundary = " + repr(max_boundary) + " ----")

    print("****** 实验记录 ******")

    print("----                                 history_params_list                                     |    history_costs_list ----")
    history_params_list = f['history_params_list'][()]
    history_costs_list = f['history_costs_list'][()]
    for (params, cost) in zip(history_params_list, history_costs_list):
        print(repr(params).replace('\n', '').replace('\r', '').replace(
            ' ', '').replace(',', ', ') + ' | ' + str(cost))

    best_params = f['best_params'][()]
    print("---- best_params = " + repr(best_params).replace('\n',
                                                            '').replace('\r', '').replace(' ', '').replace(',', ', ') + " ----")

    best_cost = f['best_cost'][()]
    print("---- best_cost = %d ----" % best_cost)

    print("----                                   best_params_list                                       |    best_costs_list ----")
    best_params_list = f['best_params_list'][()]
    best_costs_list = f['best_costs_list'][()]
    for (params, cost) in zip(best_params_list, best_costs_list):
        print(repr(params).replace('\n', '').replace('\r', '').replace(
            ' ', '').replace(',', ', ') + ' | ' + str(cost))

    last_iteration = f['last_iteration'][()]
    print("---- last_iteration = %d ----" % last_iteration)

    save_params_set = f['save_params_set'][()]
    print("---- save_params_set ----")
    for params in save_params_set:
        print(repr(params).replace('\n', '').replace(
            '\r', '').replace(' ', '').replace(',', ', '))

    load_neural_net_archive_filename = f['neural_net_archive_filename'][()]
    print("---- load_neural_net_archive_filename = " +
          load_neural_net_archive_filename + " ----")


if __name__ == '__main__':
    # 查看随机波形随机
    # min_boundary = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.3])
    # max_boundary = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.9, 0.9, 0.9, 0.9, 0.9])
    # params = np.random.uniform(min_boundary, max_boundary)
    params = np.array([0.37, 0.13, 0.16, 0.35, 0.49, 0.2, 0.52, 0.13, 0.45, 0.79])

    print(params)
    startpoint = 1
    endpoint = 0
    tf = 1
    sample_rate = 1000
    t = np.arange(0, tf, 1 / sample_rate)

    num_samples = len(params) // 2
    t_scale = (1. - params[:num_samples]).cumprod()
    t_samples = (np.concatenate(([1], t_scale[:-1])) - t_scale).cumsum() * tf
    t_samples = np.concatenate(([0.], t_samples, [tf]))
    wave_samples = params[num_samples:].cumprod(
    ) * (startpoint - endpoint) + endpoint
    wave_samples = np.concatenate(([startpoint], wave_samples, [endpoint]))

    wave = waveform_interpolate(startpoint, endpoint, tf, sample_rate, params)
    
    plt.scatter(t_samples, wave_samples)
    plt.plot(t, wave)
    plt.show()
