import os
import getopt
import sys
import numpy as np
import utilities
import learner
import simulation


class Interface():
    def __init__(self):
        # 实验参数
        self.target_cost = 0
        self.num_params = 7
        self.min_boundary = [-3., -3., -3., -3., -3., -3., -3.]
        self.max_boundary = [3., 3., 3., 3., 3., 3., 3.]
        self.startpoint = 10.0
        self.endpoint = 0.0
        self.tf = 15.71
        self.sample_rate = 5000

        # 训练参数
        self.initial_params_set_size = 20
        self.predict_good_params_set_size = 1000
        self.predict_random_params_set_size = 10000
        self.select_random_params_set_size = 20
        self.window_size = 10
        self.max_num_iteration = 100
        self.save_params_set_size = 20

        # 实验文件参数
        self.wave_dir = "./waves"
        self.signal_dir = "./signals"
        self.result_dir = "./results"
        self.init_params_index = 1
        self.params_index = 0
        self.init_result_index = 777
        self.result_index = self.init_result_index

        # 训练文件参数
        self.archive_dir = "./archives"
        self.load_archive_datetime = None

    def get_experiment_costs(self, params_set):
        return self.get_experiment_costs_test(params_set)

    def get_experiment_costs_vapor(self, params_set):
        costs = np.array([], dtype=float)
        for params in params_set:
            # 生成波形
            wave = utilities.waveform(
                self.startpoint, self.endpoint, self.tf, self.sample_rate, params)
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
                self.params_index += 1
            # 产生有效成本，结果文件序号增一
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
            if not bad:
                costs = np.hstack((costs, t - min_time))
            else:
                costs = np.hstack((costs, 1000.0))
        return costs
    
    def get_experiment_costs_simulation(self, params_set):
        costs = np.array([], dtype=float)
        for params in params_set:
            wave = utilities.waveform(
                self.startpoint, self.endpoint, self.tf, self.sample_rate, params)
            omega_r_0 = 13135.56
            omega_z_0 = 99.86535
            omega_r = omega_r_0 * np.sqrt(wave / wave[0])
            omega_z = omega_z_0 * np.sqrt(wave / wave[0])
            cost = simulation.calculate_temperature(wave, omega_r, omega_z, self.sample_rate)
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
    learn.plot_best_cost_list()


if __name__ == '__main__':
    main(sys.argv[1:])
