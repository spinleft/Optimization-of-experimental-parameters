import os
import time
import datetime
import numpy as np
import interface
import utilities
import neuralnet
import h5py
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import multiprocessing


class Learner():
    def __init__(self,
                 interface):
        self.interface = interface
        # 实验参数
        self.num_params = interface.num_params
        self.min_boundary = np.array(interface.min_boundary)
        self.max_boundary = np.array(interface.max_boundary)
        if self.num_params != len(self.min_boundary) or self.num_params != len(self.max_boundary):
            print("num_params != boundary")
            raise ValueError

        # 神经网络超参数
        self.layer_dims = [192] * 5
        # 神经网络的验证集误差下降小于train_threshold_ratio若干次时，停止训练
        self.train_threshold_ratio = 0.01
        self.batch_size = 16                    # 神经网络训练的批量大小
        self.dropout_prob = 0.6667                 # 神经元随机失效的概率
        self.regularisation_coefficient = 0.01   # loss正则化的系数
        self.max_epoch = 5000

        # 训练参数
        self.target_cost = interface.target_cost
        self.initial_params_set_size = interface.initial_params_set_size
        self.subsequent_params_set_size = interface.subsequent_params_set_size
        self.predict_good_params_set_size = interface.predict_good_params_set_size
        self.predict_random_params_set_size = interface.predict_random_params_set_size
        self.window_size = interface.window_size
        self.max_num_iteration = interface.max_num_iteration
        self.save_params_set_size = interface.save_params_set_size
        self.init_net_weight_num = 10       # 初始化神经网络时尝试随机权重的次数
        self.std_dev = 0.02                 # 生成正态分布参数的标准差（将上下界差缩放为1后）

        # 训练文件
        self.archive_dir = interface.archive_dir                    # 存档目录
        self.archive_file_prefix = 'archive_'                       # 存档前缀
        self.start_datetime = utilities.get_datetime_now_string()   # 存档日期
        self.archive_filename = os.path.join(self.archive_dir,
                                             self.archive_file_prefix+self.start_datetime+'.h5')
        # 存档
        self.archive = {'num_params': self.num_params,
                        'min_boundary': self.min_boundary,
                        'max_boundary': self.max_boundary,
                        'layer_dims': self.layer_dims,
                        'train_threshold_ratio': self.train_threshold_ratio,
                        'batch_size': self.batch_size,
                        'dropout_prob': self.dropout_prob,
                        'regularisation_coefficient': self.regularisation_coefficient
                        }

        # 创建进程池
        self.num_cores = os.cpu_count()
        self.pool = multiprocessing.Pool(processes=self.num_cores)

    def _initialize_neural_net(self):
        # 新建神经网络
        self.net = neuralnet.NeuralNet(self.min_boundary,
                                       self.max_boundary,
                                       costs_mean=0.3,
                                       costs_stdev=0.22,
                                       archive_dir=self.archive_dir,
                                       start_datetime=self.start_datetime)
        self.net.init(self.num_params,
                      self.layer_dims,
                      self.train_threshold_ratio,
                      self.batch_size,
                      self.dropout_prob,
                      self.regularisation_coefficient)
        # 随机初始化网络多次，选择在窗口参数上 loss 最小的权重
        best_loss = float('inf')
        for _ in range(self.init_net_weight_num):
            self.net.reset_weights()
            self.net.fit(self.history_params_list,
                         self.history_costs_list,
                         self.max_epoch)
            loss = self.net.get_loss(self.history_params_list, self.history_costs_list) + 3 * self.net.get_loss(self.window_params_set, self.window_costs_set)
            if loss < best_loss:
                best_loss = loss
                best_weights = self.net.get_weights()
        self.net.set_weights(best_weights)

    def _reset_neural_net(self):
        # 重置神经网络权重
        # 随机初始化网络多次，选择在窗口参数上 loss 最小的权重
        best_loss = self.net.get_loss(self.history_params_list, self.history_costs_list) + 3 * self.net.get_loss(self.window_params_set, self.window_costs_set)
        best_weights = self.net.get_weights()
        for _ in range(self.init_net_weight_num):
            self.net.reset_weights()
            self.net.fit(self.history_params_list,
                         self.history_costs_list,
                         self.max_epoch)
            loss = self.net.get_loss(self.history_params_list, self.history_costs_list) + 3 * self.net.get_loss(self.window_params_set, self.window_costs_set)
            if loss < best_loss:
                print("Better loss found...")
                best_loss = loss
                best_weights = self.net.get_weights()
        self.net.set_weights(best_weights)

    def init(self):
        # 随机产生一组参数，获取实验结果
        print("Iteration 0...")
        self.init_params_set = self.get_init_params_set()
        actual_init_costs_set, self.init_costs_set = self.get_experiment_costs(self.init_params_set)
        self.history_params_list = self.init_params_set
        self.history_costs_list = self.init_costs_set
        self.actual_costs_set = actual_init_costs_set

        # 筛选好的参数放入窗口
        indexes = np.argsort(self.init_costs_set)
        self.window_params_set = self.init_params_set[indexes[:self.window_size[0]]]
        self.window_costs_set = self.init_costs_set[indexes[:self.window_size[0]]]
        # 记录初始化的最好参数和结果
        self.best_params = self.init_params_set[indexes[0]]
        self.best_cost = self.init_costs_set[indexes[0]]
        # print("The best params in iteration 0: ")
        # print(self.best_params)
        print("The best cost in iteration 0: ")
        print(self.best_cost)
        # print("window_cost_set:")
        # print(self.window_costs_set)
        # 新建记录列表
        self.best_params_list = np.array([self.best_params], dtype=float)
        self.best_costs_list = np.array([self.best_cost], dtype=float)
        self.last_iteration = 0
        # 记入档案
        self.archive.update({'last_iteration': self.last_iteration})

        # 随机初始化神经网络，选择训练后 loss 最小的网络
        print("Initializing net...")
        self._initialize_neural_net()
        # 存档
        # self._save_archive()

    def load(self, start_datetime):
        # 加载存档
        load_archive_filename = os.path.join(
            self.archive_dir, self.archive_file_prefix+start_datetime+'.h5')
        # 从存档中读取参数
        print("Loading...")
        self._load_archive(load_archive_filename)

        # 读取上次保存的典型参数，获取实验结果
        self.last_iteration += 1
        print("Iteration %d..." % self.last_iteration)
        _, self.init_costs_set = self.get_experiment_costs(
            self.init_params_set)
        self.history_params_list = np.vstack(
            (self.history_params_list, self.init_params_set))
        self.history_costs_list = np.hstack(
            (self.history_costs_list, self.init_costs_set))

        # 筛选好的参数放入窗口，最多不超过初始参数的一半
        indexes = np.argsort(self.init_costs_set)
        self.window_params_set = self.init_params_set[indexes[:self.window_size[0]]]
        self.window_costs_set = self.init_costs_set[indexes[:self.window_size[0]]]
        # 记录典型参数中的最好参数和结果
        self.best_params = self.init_params_set[indexes[0]]
        self.best_cost = self.init_costs_set[indexes[0]]
        # 记入记录列表
        self.best_params_list = np.vstack(
            (self.best_params_list, self.best_params))
        self.best_costs_list = np.hstack(
            (self.best_costs_list, self.best_cost))
        # 更新档案
        self.archive.update({'last_iteration': self.last_iteration})
        # 加载神经网络
        self.net = neuralnet.NeuralNet(self.min_boundary,
                                       self.max_boundary,
                                       archive_dir=self.archive_dir,
                                       start_datetime=self.start_datetime)
        self.net.load(self.archive, self.load_neural_net_archive_filename)
        # 存档
        # self._save_archive()

    def close(self):
        self.pool.close()
        self.pool.join()

    def train(self):
        iteration = 1
        for i in range(self.last_iteration + 1, self.last_iteration + self.max_num_iteration):
            print("Iteration %d..." % i)
            # Step1: 训练神经网络
            history = self.net.fit(self.history_params_list,
                                self.history_costs_list,
                                self.max_epoch,
                                self.history_params_list,
                                self.history_costs_list)
            self.max_epoch += 100
            print("last loss = %f"%history.history['loss'][-1])
            print("training epoches = %d" % len(history.epoch))
            # 测量神经网络拟合误差
            fit_history_costs_set = self.net.predict_costs(self.history_params_list)
            fit_loss = np.average(np.abs(self.history_costs_list - fit_history_costs_set))
            print("fit_loss = %f" %fit_loss)
            # Step2: 产生预测参数
            # index = (iteration - 1) % len(self.predict_good_params_set_size)
            index = 0
            predict_good_params_set = []
            for window_params in self.window_params_set:
                predict_good_params_set.append(self.get_predict_good_params_set(window_params, self.predict_good_params_set_size[index]))
            predict_good_params_set = np.vstack(predict_good_params_set)
            predict_random_params_set = self.get_predict_random_params_set(self.predict_random_params_set_size[index])

            # Step3: 选出下一次实验的参数
            predict_params_set = np.vstack((predict_good_params_set, predict_random_params_set))
            predict_costs_set = self.net.predict_costs(predict_params_set)
            indexes = np.argsort(predict_costs_set)
            select_params_set = []
            for index in indexes:
                if utilities.params_in_condition(predict_params_set[index]):
                    select_params_set.append(predict_params_set[index])
                    if len(select_params_set) >= self.subsequent_params_set_size:
                        break
            select_params_set = np.vstack(select_params_set)

            # Step4: 获取实验结果
            actual_select_costs_set, select_costs_set = self.get_experiment_costs(select_params_set)
            self.history_params_list = np.vstack((self.history_params_list, select_params_set))
            self.history_costs_list = np.hstack((self.history_costs_list, select_costs_set))
            self.actual_costs_set = np.hstack((self.actual_costs_set, actual_select_costs_set))
            # 测量神经网络预测误差
            predict_select_costs_set = self.net.predict_costs(select_params_set)
            predict_loss = np.average(np.abs(actual_select_costs_set - predict_select_costs_set))
            print("predict_loss = %f" %predict_loss)
            # 得到新的窗口
            indexes = np.argsort(self.history_costs_list)
            self.window_params_set = self.history_params_list[indexes[:self.window_size[iteration]]]
            iteration += 1

            # 记录训练结果
            index = np.argmin(select_costs_set)
            iteration_best_params = select_params_set[index]
            iteration_best_cost = select_costs_set[index]
            # 更新本次训练的最好参数和结果
            if iteration_best_cost < self.best_cost:
                self.best_params = iteration_best_params
                self.best_cost = iteration_best_cost
            # 将本次循环的最好参数和结果加入列表
            self.best_params_list = np.vstack((self.best_params_list, iteration_best_params))
            self.best_costs_list = np.hstack((self.best_costs_list, iteration_best_cost))
            # 更新档案
            self.archive.update({'last_iteration': i})
            # 存档
            # self._save_archive()
            # print("The best params in iteration %d: " % i)
            # print(iteration_best_params)
            print("The best cost in iteration %d: " % i)
            print(iteration_best_cost)
            # print("window_cost_set:")
            # print(self.window_costs_set)

        print("The best parameters: " + str(self.best_params))
        print("The best cost: " + str(self.best_cost))
        # self._save_archive()

    def get_init_params_set(self):
        # 并行产生随机数
        block_size = self.initial_params_set_size // self.num_cores
        blocks = [block_size] * (self.num_cores - 1) + \
            [self.initial_params_set_size - block_size * (self.num_cores - 1)]
        multiple_results = [self.pool.apply_async(utilities.get_init_params_set, args=(
            self.min_boundary,
            self.max_boundary,
            params_set_size
        )) for params_set_size in blocks]
        params_set_list = [result.get() for result in multiple_results]
        params_set = params_set_list[0]
        for i in range(1, self.num_cores):
            params_set = np.vstack((params_set, params_set_list[i]))
        return params_set

    def get_predict_good_params_set(self, base_params, params_set_size):
        # 并行产生随机数
        block_size = params_set_size // self.num_cores
        blocks = [block_size] * (self.num_cores - 1) + [params_set_size - block_size * (self.num_cores - 1)]
        multiple_results = [self.pool.apply_async(utilities.get_normal_params_set, args=(
            self.min_boundary,
            self.max_boundary,
            base_params,
            self.std_dev,
            params_set_size
        )) for params_set_size in blocks]
        params_set_list = [result.get() for result in multiple_results]
        params_set = params_set_list[0]
        for i in range(1, self.num_cores):
            params_set = np.vstack((params_set, params_set_list[i]))
        return params_set

    def get_predict_random_params_set(self, params_set_size):
        # 并行产生随机数
        block_size = params_set_size // self.num_cores
        blocks = [block_size] * (self.num_cores - 1) +  [params_set_size - block_size * (self.num_cores - 1)]
        multiple_results = [self.pool.apply_async(utilities.get_random_params_set, args=(
            self.min_boundary,
            self.max_boundary,
            params_set_size
        )) for params_set_size in blocks]
        params_set_list = [result.get() for result in multiple_results]
        params_set = params_set_list[0]
        for i in range(1, self.num_cores):
            params_set = np.vstack((params_set, params_set_list[i]))
        return params_set

    def get_experiment_costs(self, params_set):
        # 并行实现
        # multiple_results = [self.pool.apply_async(self.interface.get_experiment_costs, args=(
        #     np.array([params]),)) for params in params_set]
        # costs_list = [result.get() for result in multiple_results]
        # costs = np.array(costs_list).reshape(-1,)
        costs = self.interface.get_experiment_costs(params_set)
        return costs

    def _save_archive(self):
        save_params_set = None
        # K-聚类获得典型参数
        # 构造聚类器
        clusters = min(self.save_params_set_size, len(self.history_params_list))
        self.k_means = KMeans(
            n_clusters=clusters, max_iter=1000)
        # 聚类后取每一类中结果最好的参数
        self.k_means.fit(self.history_params_list)
        labels = self.k_means.predict(self.history_params_list)
        for i in range(clusters):
            params_subset = self.history_params_list[labels == i]
            costs_subset = self.history_costs_list[labels == i]
            index = np.argmin(costs_subset)
            if save_params_set is None:
                save_params_set = np.array([params_subset[index]])
            else:
                save_params_set = np.vstack(
                    (save_params_set, params_subset[index]))
        self.archive.update({'history_params_list': self.history_params_list,
                             'history_costs_list': self.history_costs_list,
                             'best_params_list': self.best_params_list,
                             'best_costs_list': self.best_costs_list,
                             'best_params': self.best_params,
                             'best_cost': self.best_cost,
                             'save_params_set': save_params_set,
                             'neural_net_archive_filename': self.net.save()})
        f = h5py.File(self.archive_filename, 'w')
        for key in self.archive:
            f.create_dataset(key, data=self.archive[key])
        f.close()

    def _load_archive(self, archive_filename):
        f = h5py.File(archive_filename, 'r')
        # 实验参数
        num_params = f['num_params'][()]    # 参数数量
        if self.num_params is not None and self.num_params != num_params:
            print("self.num_params != num_params")
            raise ValueError
        else:
            self.num_params = num_params
        min_boundary = f['min_boundary'][()]   # 参数下界
        if self.min_boundary is not None and (self.min_boundary != min_boundary).any():
            print("self.min_boundary != min_boundary")
            raise ValueError
        else:
            self.min_boundary = min_boundary
        max_boundary = f['max_boundary'][()]   # 参数上界
        if self.max_boundary is not None and (self.max_boundary != max_boundary).any():
            print("self.max_boundary != max_boundary")
            raise ValueError
        else:
            self.max_boundary = max_boundary

        # 实验记录
        self.history_params_list = f['history_params_list'][()]
        self.history_costs_list = f['history_costs_list'][()]
        self.best_params = f['best_params'][()]
        self.best_cost = f['best_cost'][()]
        self.best_params_list = f['best_params_list'][()]
        self.best_costs_list = f['best_costs_list'][()]
        self.last_iteration = f['last_iteration'][()]
        self.init_params_set = f['save_params_set'][()]
        self.load_neural_net_archive_filename = f['neural_net_archive_filename'][(
        )]

    def plot_best_costs_list(self):
        x_axis = np.arange(start=0, stop=len(
            self.best_costs_list), step=1, dtype=int)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.plot(x_axis, self.best_costs_list)
        plt.show()

    def print_archive(self):
        f = h5py.File(self.archive_filename, 'r')
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
