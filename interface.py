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
        self.num_params = 10
        self.min_boundary = [0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.max_boundary = [0.35, 0.35, 0.35, 0.35, 0.35, 0.9, 0.9, 0.9, 0.9, 0.9]
        self.patch_length = 0.01
        self.startpoint = 10.
        self.endpoint = 0.
        self.tf = 15.71
        self.sample_rate = 5000                             # 实验采样率
        self.uncer = 0.45

        # 训练参数
        self.target_cost = 0
        self.max_num_iteration = 100                         # 最大迭代次数
        self.initial_params_set_size = 20                   # 初始实验数量
        self.subsequent_params_set_size = 20
        self.window_size = 10
        self.predict_good_params_set_size = 100000
        self.predict_random_params_set_size = [i**2 * 50000 for i in range(5)]
        self.save_params_set_size = 20                      # 存档中保存的典型参数数量

        # 实验文件参数
        self.wave_dir = "//192.168.0.134/Share/mlparams/waveform"               # 波形文件目录
        self.tf_filename = "//192.168.0.134/Share/mlparams/waveform/tf.txt"     # 终止时间文件名
        self.signal_dir = "//192.168.0.134/Share/mlparams/index"                # 实验信号文件目录
        self.result_dir = "./results"                                           # 实验结果目录
        # 信号文件初始序号
        self.signal_index = 1
        self.init_result_index = 183                                            # 初始结果序号
        self.result_index = self.init_result_index

        # 训练文件参数
        self.archive_dir = "./archives"
        self.load_archive_datetime = None

    def params_in_condition(self, params):
        # 限制初始斜率
        # a_1 = -1 - np.sum(params)
        # coef = np.hstack((a_1, params))
        # l = int((len(params) + 1) / 2)
        # slope = coef[0]
        # for i in range(l):
        #     slope += (np.power(2, 2*i+3) - 1) / (2 * i + 3) * coef[l+i]
        # if slope <= 0 and abs(slope) < 50:
        #     # 限制上下限
        #     wave = waveform_polyn(1, 0, 1, 80000, params)
        #     if np.max(wave) <= 1:
        #         return True
        #     else:
        #         return False
        # else:
        #     return False
        return True

    def get_experiment_costs(self, params_set):
        return self.get_experiment_costs_test(params_set)

    def get_experiment_costs_ramp(self, params_set):
        costs = np.array([], dtype=float)
        for params in params_set:
            # 生成波形
            wave1 = utilities.waveform_linear(
                params[0], params[1], self.tf, self.sample_rate)
            wave2 = utilities.waveform_linear(
                params[2], params[3], self.tf, self.sample_rate)
            # wave3 = utilities.waveform_linear(params[4], params[5], self.tf, self.sample_rate)
            # wave4 = utilities.waveform_linear(params[0], params[7], self.tf, self.sample_rate)
            # 保存波形到文件
            wave1_filename = os.path.join(self.wave_dir, '1.txt')
            wave2_filename = os.path.join(self.wave_dir, '2.txt')
            # wave3_filename = os.path.join(self.wave_dir, '3.txt')
            # wave4_filename = os.path.join(self.wave_dir, '4.txt')
            const3_filename = os.path.join(self.wave_dir, '3.txt')
            const4_filename = os.path.join(self.wave_dir, '4.txt')
            const5_filename = os.path.join(self.wave_dir, '5.txt')
            const6_filename = os.path.join(self.wave_dir, '6.txt')

            utilities.save_params_to_file(wave1_filename, wave1)
            utilities.save_params_to_file(wave2_filename, wave2)
            # utilities.save_params_to_file(wave3_filename, wave3)
            # utilities.save_params_to_file(wave4_filename, wave4)
            utilities.save_params_to_file(const3_filename, [params[4]])
            utilities.save_params_to_file(const4_filename, [params[5]])
            utilities.save_params_to_file(const5_filename, [params[6]])
            utilities.save_params_to_file(const6_filename, [params[7]])
            # 发送信号文件
            signal_filename = os.path.join(
                self.signal_dir, str(self.signal_index) + '.txt')
            utilities.save_params_to_file(signal_filename, [])
            # 信号文件序号增一
            self.signal_index += 1
            # 读取实验结果
            result_filename = os.path.join(
                self.result_dir, str(self.result_index) + '.txt')
            temp = utilities.get_result_from_file(result_filename)
            # 计算cost
            bad = False
            cost = temp[0]
            if temp[1] < 1e6:
                bad = True
            # 产生结果，结果文件序号增一
            self.result_index += 1
            while bad == True:
                # 失锁等原因产生坏数据，重新进行实验
                # 保存波形到文件
                utilities.save_params_to_file(wave1_filename, wave1)
                utilities.save_params_to_file(wave2_filename, wave2)
                # utilities.save_params_to_file(wave3_filename, wave3)
                # utilities.save_params_to_file(wave4_filename, wave4)
                utilities.save_params_to_file(const3_filename, [params[4]])
                utilities.save_params_to_file(const4_filename, [params[5]])
                utilities.save_params_to_file(const5_filename, [params[6]])
                utilities.save_params_to_file(const6_filename, [params[7]])
                # 发送信号文件
                signal_filename = os.path.join(
                    self.signal_dir, str(self.signal_index) + '.txt')
                utilities.save_params_to_file(signal_filename, [])
                # 参数文件序号增一
                self.signal_index += 1
                # 读取实验结果
                result_filename = os.path.join(
                    self.result_dir, str(self.result_index) + '.txt')
                temp = utilities.get_result_from_file(result_filename)
                # 计算cost
                cost = temp[0]
                self.result_index += 1
                # bad = ...
                if temp[1] > 1e6:
                    bad = False
            print("atom num = %f, cost = %.5f" % (temp[1], temp[0]))
            costs = np.hstack((costs, cost))
        return costs

    def get_experiment_costs_vapor(self, params_set):
        costs = np.array([], dtype=float)
        for params in params_set:
            # utilities.plot_wave(self.startpoint, self.endpoint, params[-1], self.sample_rate, params[:-1])
            # 生成波形
            wave = utilities.waveform_polyn(
                self.startpoint, self.endpoint, params[-1], self.sample_rate, params[:-1])
            # 保存波形到文件
            wave_filename = os.path.join(self.wave_dir, 'dipole1evp.txt')

            utilities.save_params_to_file(wave_filename, wave)
            # utilities.save_params_to_file(self.tffilename, [round(params[-1]*1e6)])
            # 发送信号文件
            signal_filename = os.path.join(
                self.signal_dir, str(self.signal_index) + '.txt')
            utilities.save_params_to_file(signal_filename, [])
            # 参数文件序号增一
            self.signal_index += 1
            # 读取实验结果
            result_filename = os.path.join(
                self.result_dir, str(self.result_index) + '.txt')
            temp = utilities.get_result_from_file(result_filename)
            # 计算cost
            bad = False
            cost = temp[0]
            if temp[1] < 1e5:
                bad = True
            # bad = ...
            # 产生结果，结果文件序号增一
            self.result_index += 1
            while bad == True:
                # 失锁等原因产生坏数据，重新进行实验
                # 保存波形到文件
                wave_filename = os.path.join(self.wave_dir, 'dipole1evp.txt')
                utilities.save_params_to_file(wave_filename, wave)
                # utilities.save_params_to_file(self.tffilename, [round(params[-1]*1e6)])
                # 发送信号文件
                signal_filename = os.path.join(
                    self.signal_dir, str(self.signal_index) + '.txt')
                utilities.save_params_to_file(signal_filename, [])
                # 参数文件序号增一
                self.signal_index += 1
                # 读取实验结果
                result_filename = os.path.join(
                    self.result_dir, str(self.result_index) + '.txt')
                temp = utilities.get_result_from_file(result_filename)
                # 计算cost
                cost = temp[0]
                self.result_index += 1
                # bad = ...
                if temp[1] > 1e5:
                    bad = False
            print("atom num = %f, cost = %.5f" % (temp[1], temp[0]))
            costs = np.hstack((costs, cost))
        return costs

    def get_experiment_costs_test(self, params_set):
        actual_costs = []
        costs = []
        for params in params_set:
            print(repr(params).replace('\n', '').replace(
                '\r', '').replace(' ', '').replace(',', ', '))
            k = 5.0
            g = 9.8
            x_step = 1.0 / self.sample_rate
            xmax = 15.71
            x = np.arange(0, xmax, x_step)
            len_x = len(x)
            t = 0
            bad = False
            wave = utilities.waveform_interpolate(
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
            if not bad:
                actual_cost = t - min_time
                actual_costs.append(actual_cost)
                t += t * np.random.normal(0, 0.1)
                cost = t - min_time
                costs.append(cost)
            else:
                actual_cost = 10.
                cost = 10.
                actual_costs.append(actual_cost)
                costs.append(actual_cost)
            print("actual_cost = %f, cost = %f" % (actual_cost, cost))
        actual_costs = np.array(actual_costs)
        costs = np.array(costs)
        return (actual_costs, costs)

    def get_experiment_costs_simulation(self, params_set):
        costs = np.array([], dtype=float)
        for params in params_set:
            wave = utilities.waveform_polyn(
                self.startpoint, self.endpoint, self.tf, self.sample_rate, params)
            cost = simulation.calculate_temperature(wave, self.sample_rate)
            cost += cost * np.random.normal(0, 0.1)
            cost = np.log(cost / 0.084)
            costs = np.hstack((costs, cost))
        return costs


def main(argv):
    load = False
    if len(argv) != 0:
        option = argv[0]
        value = argv[1]
        if option in ("-h", "--help"):
            print("initialize a new experiment: run interface.py with no args")
            print(
                "continue experiments based on the archive: interface.py -l[--load] \"YYYY-MM-DD_hh-mm\"")
            return
        elif option in ("-l", "--load"):
            load = True
            datetime = value
        elif option in ("-p", "--print"):
            archive_filename = './archives/archive_' + value + '.h5'
            utilities.print_archive(archive_filename)
            return
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
