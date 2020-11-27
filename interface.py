import os
import getopt
import sys
import numpy as np
import utilities
import learner
from scipy import constants
import simulation


class Interface():
    def __init__(self):
        # 实验参数
        self.target_cost = 0
        self.num_params = 7
        self.min_boundary = [-3., -3., -3., -3., -3., -3., -3.]
        self.max_boundary = [3., 3., 3., 3., 3., 3., 3.]
        self.startpoint = 12 * constants.Boltzmann * 1.5e-6
        self.endpoint = self.startpoint / 25
        self.tf = 10
        self.sample_rate = 20

        # 训练参数
        self.initial_params_set_size = 20           # 初始实验数量
        self.predict_good_params_set_size = 100     # 每次迭代，以窗口中每个参数为均值生成正态分布参数数量，选择一个作为下一次实验参数
        self.predict_random_params_set_size = 1000  # 每次迭代，生成均匀分布参数数量
        self.select_random_params_set_size = 10     # 每次迭代，选择均匀分布参数数量，作为下一次实验参数
        self.window_size = 10                       # 窗口最大大小
        self.max_num_iteration = 100                # 最大迭代次数
        self.save_params_set_size = 20              # 存档中保存的典型参数数量

        # 实验文件参数
        self.wave_dir = "./waves"                   # 波形文件目录
        self.signal_dir = "./signals"               # 实验信号文件目录
        self.result_dir = "./results"               # 实验结果目录
        self.params_index = 0
        self.init_result_index = 777                # 初始结果序号
        self.result_index = self.init_result_index

        # 训练文件参数
        self.archive_dir = "./archives"
        self.load_archive_datetime = None

    def get_experiment_costs(self, params_set):
        return self.get_experiment_costs_simulation(params_set)

    def get_experiment_costs_vapor(self, params_set):
        costs = np.array([], dtype=float)
        for params in params_set:
            # 生成波形
            wave = utilities.waveform(
                self.startpoint, self.endpoint, params[-1], self.sample_rate, params[:-1])
            # 保存波形到文件
            wave_filename = os.path.join(
                self.wave_dir, str(self.params_index)+'.txt')
            utilities.save_params_to_file(wave_filename, wave)
            # 发送信号文件
            signal_filename = os.path.join(
                self.signal_dir, str(self.params_index)+'.txt')
            utilities.save_params_to_file(signal_filename, [])
            # 参数文件序号增一
            self.params_index += 1
            # 读取实验结果
            result_filename = os.path.join(
                self.result_dir, str(self.result_index)+'.txt')
            temp = utilities.get_result_from_file(result_filename)
            # 结果文件序号增一，准备读取下一个结果
            self.result_index += 1
            # 计算cost
            bad = False
            cost = temp
            # bad = ...
            while bad == True:
                # 失锁等原因产生坏数据，重新进行实验
                # 保存波形到文件
                wave_filename = os.path.join(
                    self.wave_dir, str(self.params_index)+'.txt')
                utilities.save_params_to_file(wave_filename, wave)
                # 发送信号文件
                signal_filename = os.path.join(
                    self.signal_dir, str(self.params_index)+'.txt')
                utilities.save_params_to_file(signal_filename, [])
                # 参数文件序号增一
                self.params_index += 1
                # 读取实验结果
                result_filename = os.path.join(
                    self.result_dir, str(self.result_index)+'.txt')
                temp = utilities.get_result_from_file(result_filename)
                # 计算cost
                cost = temp
                # bad = ...
                self.result_index += 1
            
            costs = np.hstack((costs, cost))
        return costs

    def get_experiment_costs_test(self, params_set):
        costs = np.array([], dtype=float)
        for params in params_set:
            k = 5.0
            g = 9.8
            x_step = 1.0 / self.sample_rate
            xmax = 15.71
            x = np.arange(0, xmax, x_step)
            len_x = len(x)
            t = 0
            bad = False
            wave = utilities.waveform(
                self.startpoint, self.endpoint, self.tf, self.sample_rate, params)
            for i in range(1, len_x):
                if wave[i] > 10.0 or wave[i] == float('nan'):
                    bad = True
                    break
                else:
                    v_i = np.sqrt(2 * g * (10.0 - wave[i - 1]))
                    s = np.sqrt((x[i] - x[i - 1]) ** 2 +
                                (wave[i - 1] - wave[i]) ** 2)
                    a = (wave[i - 1] - wave[i]) / s
                    if np.abs(a) < 1e-15:
                        t += s / v_i
                    else:
                        t += (np.sqrt(v_i ** 2 + 2 * a * s) - v_i) / a
            min_time = np.pi * np.sqrt(k / g)
            t += t * np.random.normal(0, 0.1)
            cost = t - min_time
            if not bad:
                costs = np.hstack((costs, cost))
            else:
                costs = np.hstack((costs, 1000.0))
        return costs

    def get_experiment_costs_simulation(self, params_set):
        costs = np.array([], dtype=float)
        for params in params_set:
            wave = utilities.waveform(
                self.startpoint, self.endpoint, self.tf, self.sample_rate, params)
            cost = simulation.calculate_temperature(wave, self.sample_rate)
            cost = np.log(cost / 0.08)
            costs = np.hstack((costs, cost))
        return costs


def main(argv):
    try:
        options, _ = getopt.getopt(argv, "hl:", ["help", "load="])
    except getopt.GetoptError:
        sys.exit()
    load = False
    for option, value in options:
        if option in ("-h", "--help"):
            print("initialize a new experiment: run interface.py with no args")
            print(
                "continue experiments based on the archive: interface.py -l[--load] \"YYYY-MM-DD_hh-mm\"")
        if option in ("-l", "--load"):
            load = True
            datetime = value

    interface = Interface()
    learn = learner.Learner(interface)
    if load:
        learn.load(datetime)
    else:
        learn.init()
    learn.train()
    learn.plot_best_costs_list()
    learn.close()


if __name__ == '__main__':
    main(sys.argv[1:])
