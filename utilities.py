import os
import datetime
import numpy as np
import tensorflow as tf


def get_dict_from_file(filename):
    '''
    Method for getting a dictionary from a file, of a given format. 

    Args:    
        filename (str): The filename for the file.

    Returns:
        dict : Dictionary of values in file.
    '''
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
    tdict = eval('dict(' + tdict_string + ')')
    return tdict


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


def dict_to_txt_file(tdict, filename):
    '''
    Method for writing a dict to a file with syntax similar to how files are input.

    Args:
        tdict (dict): Dictionary to be written to file.
        filename (string): Filename for file. 
    '''
    with open(filename, 'w') as out_file:
        for key in tdict:
            out_file.write(
                str(key) + '=' + repr(tdict[key]).replace('\n', '').replace('\r', '') + '\n')


def txt_file_to_dict(filename):
    '''
    Method for taking a file and changing it to a dict. Every line in file is a new entry for the dictionary and each element should be written as::

        [key] = [value]

    White space does not matter.

    Args:
        filename (string): Filename of file.

    Returns:
        dict : Dictionary of values in file. 
    '''
    with open(filename, 'r') as in_file:
        tdict_string = ''
        for line in in_file:
            temp = (line.partition('#')[0]).strip('\n').strip()
            if temp != '':
                tdict_string += temp+','
    # Setting up words for parsing a dict, ignore eclipse warnings
    array = np.array  # @UnusedVariable
    inf = float('inf')  # @UnusedVariable
    nan = float('nan')  # @UnusedVariable
    tdict = eval('dict('+tdict_string+')')
    return tdict


def get_init_params(min_boundary, max_boundary, initial_params_set_size):
    num_params = len(min_boundary)
    params_set = np.random.uniform(
        min_boundary, max_boundary, (initial_params_set_size, num_params))
    return params_set


def get_pred_params(min_boundary, max_boundary, window_params_set, predict_params_set_size, gaussian_ratio, gaussian_sigma):
    num_params = len(min_boundary)
    num_gaussian = int(gaussian_ratio * predict_params_set_size)
    num_uniform = int(predict_params_set_size - num_gaussian)
    num_window = len(window_params_set)
    num_gaussian_scale = int(np.ceil(num_gaussian / num_window))

    gaussian_loc = np.array([window_params_set for _ in range(num_gaussian_scale)]).reshape((num_window*num_gaussian_scale, -1))
    gaussian_loc = gaussian_loc[0:num_gaussian]
    gaussian_scale = gaussian_sigma * (np.array(max_boundary) - np.array(min_boundary))
    gaussian_params_set = np.random.normal(gaussian_loc, gaussian_scale, (num_gaussian, num_params))
    cond = gaussian_params_set >= min_boundary
    gaussian_params_set = np.where(cond, gaussian_params_set, min_boundary)
    cond = gaussian_params_set <= max_boundary
    gaussian_params_set = np.where(cond, gaussian_params_set, max_boundary)

    uniform_params_set = np.random.uniform(min_boundary, max_boundary, (num_uniform, num_params))
    params_set = np.vstack((gaussian_params_set, uniform_params_set))
    return params_set

    
if __name__ == '__main__':
    tdict = {'a': 'neural_net', 'b': np.array([1,2,3,4,5]), 'c': {1: 2, 3: 4}}
    dict_to_txt_file(tdict, './test.txt')
    fdict = txt_file_to_dict('./test.txt')
    print(fdict)
