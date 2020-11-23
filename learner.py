import os
import datetime
import numpy as np
import interface
import utilities
import neuralnet
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class Learner():
    def __init__(self,
                 interface):
        self.interface = interface
        # 实验参数
        self.num_params = interface.num_params
        self.min_boundary = np.array(interface.min_boundary)
        self.max_boundary = np.array(interface.max_boundary)

        # 神经网络超参数
        self.layer_dims = [64] * 5
        # 神经网络的loss下降小于train_threshold_ratio若干次时，停止训练
        self.train_threshold_ratio = 0.01
        self.batch_size = 16                    # 神经网络训练的批量大小
        self.dropout_prob = 0.5                 # 神经元随机失效的概率
        self.regularisation_coefficient = 1e-8  # loss正则化的系数
        self.max_epoch = 1000

        # 训练参数
        self.initial_params_set_size = interface.initial_params_set_size
        self.predict_good_params_set_size = interface.predict_good_params_set_size
        self.predict_random_params_set_size = interface.predict_random_params_set_size
        self.select_random_params_set_size = interface.select_random_params_set_size
        self.window_size = interface.window_size
        self.max_num_iteration = interface.max_num_iteration
        self.save_params_set_size = interface.save_params_set_size
        self.init_net_weight_num = 10
        self.std_dev = 0.03

        # 训练文件
        self.archive_dir = interface.archive_dir
        self.archive_file_prefix = 'archive_'
        self.start_datetime = utilities.get_datetime_now_string()
        self.archive_filename = os.path.join(self.archive_dir,
                                             self.archive_file_prefix+self.start_datetime+'.txt')
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

        # 构造聚类器
        self.k_means = KMeans(
            n_clusters=self.save_params_set_size, max_iter=1000)

    def initialize_neural_net(self):
        # 新建神经网络
        self.net = neuralnet.NeuralNet(self.min_boundary,
                                  self.max_boundary,
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
            self.net.fit(self.train_params_set,
                    self.train_costs_set,
                    self.max_epoch)
            loss = self.net.get_loss(self.window_params_set, self.window_costs_set)
            if loss < best_loss:
                best_loss = loss
                best_weights = self.net.get_weights()
        self.net.set_weights(best_weights)
        self.last_neural_net_weight = best_weights

    def reset_neural_net(self):
        # 随机初始化网络多次，选择在窗口参数上 loss 最小的权重
        best_loss = float('inf')
        for _ in range(self.init_net_weight_num):
            self.net.fit(self.train_params_set,
                    self.train_costs_set,
                    self.max_epoch)
            loss = self.net.get_loss(self.window_params_set, self.window_costs_set)
            if loss < best_loss:
                best_loss = loss
                best_weights = self.net.get_weights()
            self.net.reset_weights()
        self.net.set_weights(best_weights)

    def init(self):
        # 随机产生一组参数，获取实验结果
        self.train_params_set = utilities.get_init_params(
            self.min_boundary, self.max_boundary, self.initial_params_set_size)
        self.train_costs_set = self.interface.get_experiment_costs(
            self.train_params_set)

        # 将全部或部分最好的参数放入窗口
        self.window_params_set = self.train_params_set
        self.window_costs_set = self.train_costs_set
        if self.window_size < len(self.window_costs_set):
            indexes = np.argsort(self.window_costs_set)
            self.window_params_set = self.window_params_set[indexes[:self.window_size]]
            self.window_costs_set = self.window_costs_set[indexes[:self.window_size]]

        # 随机初始化神经网络，选择训练后 loss 最小的网络
        print("Initializing.")
        self.initialize_neural_net()

        # 记录初始化的最好参数和结果
        index = np.argmin(self.train_costs_set)
        self.best_params = self.train_params_set[index]
        self.best_cost = self.train_costs_set[index]
        # 新建记录列表
        self.best_params_list = np.array([self.best_params], dtype=float)
        self.best_costs_list = np.array([self.best_cost], dtype=float)
        self.last_iteration = 0
        # 记入档案
        self.archive.update({'best_params_list': self.best_params_list,
                             'best_costs_list': self.best_cost,
                             'best_params': self.best_params,
                             'best_cost': self.best_cost,
                             'last_iteration': self.last_iteration})
        # self._save_archive()

    def load(self, start_datetime):
        # 加载存档
        load_archive_filename = os.path.join(
            self.archive_dir, self.archive_file_prefix+start_datetime+'.txt')
        # 从存档中读取参数
        self.archive = utilities.get_dict_from_file(load_archive_filename)
        print("Loading.")
        # 实验参数
        num_params = int(self.archive['num_params'])
        if self.num_params is not None and self.num_params != num_params:
            print("self.num_params != num_params")
            raise ValueError
        else:
            self.num_params = num_params
        min_boundary = np.array(self.archive['min_boundary'])
        if self.min_boundary is not None and (self.min_boundary != min_boundary).any():
            print("self.min_boundary != min_boundary")
            raise ValueError
        else:
            self.min_boundary = min_boundary
        max_boundary = np.array(self.archive['max_boundary'])
        if self.max_boundary is not None and (self.max_boundary != max_boundary).any():
            print("self.max_boundary != max_boundary")
            raise ValueError
        else:
            self.max_boundary = max_boundary
        # 实验记录
        self.best_params = self.archive['best_params']
        self.best_cost = self.archive['best_cost']
        self.best_params_list = self.archive['best_params_list']
        self.best_costs_list = self.archive['best_costs_list']
        self.last_iteration = self.archive['last_iteration']

        # 加载神经网络
        load_neural_net_archive_filename = self.archive['neural_net_archive_filename']
        self.net = neuralnet.NeuralNet(self.min_boundary,
                                       self.max_boundary,
                                       archive_dir=self.archive_dir,
                                       start_datetime=self.start_datetime)
        self.net.load(self.archive, load_neural_net_archive_filename)
        # 读取上次保存的典型参数，获取实验结果
        self.train_params_set = np.array(self.archive['save_params_set'])
        self.train_costs_set = self.interface.get_experiment_costs(
            self.train_params_set)

        # 将全部或部分最好的参数放入窗口
        self.window_params_set = self.train_params_set
        self.window_costs_set = self.train_costs_set
        if self.window_size < len(self.train_costs_set):
            indexes = np.argsort(self.window_costs_set)
            self.window_params_set = self.window_params_set[indexes[:self.window_size]]
            self.window_costs_set = self.window_costs_set[indexes[:self.window_size]]

        # 记录典型参数中的最好参数和结果
        indexes = np.argsort(self.train_costs_set)
        self.best_params = self.train_params_set[indexes[0]]
        self.best_cost = self.train_costs_set[indexes[0]]
        # 记入记录列表
        self.best_params_list = np.vstack(
            (self.best_params_list, self.best_params))
        self.best_costs_list = np.hstack(
            (self.best_costs_list, self.best_cost))
        # 更新档案
        self.archive.update({'best_params_list': self.best_params_list,
                             'best_costs_list': self.best_costs_list,
                             'best_params': self.best_params,
                             'best_cost': self.best_cost,
                             'last_iteration': self.last_iteration})
        # self._save_archive()

    def train(self):

        for i in range(self.last_iteration + 1, self.last_iteration + 1 + self.max_num_iteration):
            print("Iteration %d." % i)
            # Step1: 训练神经网络
            self.net.fit(self.train_params_set,
                         self.train_costs_set,
                         self.max_epoch)

            # Step2: 产生预测参数并预测结果
            predict_good_params_sets = []
            predict_good_costs_sets = []
            for j in range(len(self.window_params_set)):
                predict_good_params_sets.append(
                    utilities.get_predict_good_params_set(
                        self.min_boundary,
                        self.max_boundary,
                        self.window_params_set[j],
                        self.predict_good_params_set_size,
                        self.std_dev))
                predict_good_costs_sets.append(
                    np.array(self.net.predict_costs(predict_good_params_sets[j])).flatten())

            predict_random_params_set = utilities.get_predict_random_params_set(
                self.min_boundary,
                self.max_boundary,
                self.predict_random_params_set_size)
            predict_random_costs_set = np.array(
                self.net.predict_costs(predict_random_params_set)).flatten()
            # Step3: 选出下一次实验的参数
            select_good_params_set = []
            for j in range(len(self.window_params_set)):
                index = np.argmin(predict_good_costs_sets[j])
                select_good_params_set.append(
                    predict_good_params_sets[j][index])
            select_good_params_set = np.array(select_good_params_set)

            indexes = np.argsort(predict_random_costs_set)
            select_random_params_set = np.array(
                predict_random_params_set[indexes[:self.select_random_params_set_size]])
            # Step4: 获取实验结果
            select_good_costs_set = self.interface.get_experiment_costs(
                select_good_params_set)
            select_random_costs_set = self.interface.get_experiment_costs(
                select_random_params_set)
            self.train_params_set = np.vstack(
                (self.train_params_set, select_good_params_set, select_random_params_set))
            self.train_costs_set = np.hstack(
                (self.train_costs_set, select_good_costs_set, select_random_costs_set))

            # 更新窗口
            cond = select_good_costs_set < self.window_costs_set
            self.window_params_set = np.where(
                cond.reshape(-1, 1), select_good_params_set, self.window_params_set)
            self.window_costs_set = np.where(
                cond, select_good_costs_set, self.window_costs_set)
            for j in range(len(select_random_params_set)):
                for k in range(len(self.window_params_set)):
                    distance = np.abs(select_random_params_set[j] - self.window_params_set[k]) / (
                        self.max_boundary - self.min_boundary)
                    if (distance < self.std_dev).all():
                        if select_random_costs_set[j] < self.window_costs_set[k]:
                            self.window_params_set[k] = select_random_params_set[j]
                            self.window_costs_set[k] = select_random_costs_set[j]
                        break
                    else:
                        self.window_params_set = np.vstack(
                            (self.window_params_set, select_random_params_set[j]))
                        self.window_costs_set = np.hstack(
                            (self.window_costs_set, select_random_costs_set[j]))
                        break
            if self.window_size < len(self.window_costs_set):
                indexes = np.argsort(self.window_costs_set)
                self.window_params_set = self.window_params_set[indexes[:self.window_size]]
                self.window_costs_set = self.window_costs_set[indexes[:self.window_size]]

            # 记录训练结果
            temp_params_set = np.vstack(
                (select_good_params_set, select_random_params_set))
            temp_costs_set = np.hstack(
                (select_good_costs_set, select_random_costs_set))
            index = np.argmin(temp_costs_set)
            iteration_best_params = temp_params_set[index]
            iteration_best_cost = temp_costs_set[index]
            # 更新本次训练的最好参数和结果
            if iteration_best_cost < self.best_cost:
                self.best_params = iteration_best_params
                self.best_cost = iteration_best_cost
            # 将本次循环的最好参数和结果加入列表
            self.best_params_list = np.vstack(
                (self.best_params_list, iteration_best_params))
            self.best_costs_list = np.hstack(
                (self.best_costs_list, iteration_best_cost))
            # 更新档案
            self.archive.update({'best_params_list': self.best_params_list,
                                 'best_costs_list': self.best_costs_list,
                                 'best_params': self.best_params,
                                 'best_cost': self.best_cost,
                                 'last_iteration': i})
            # self._save_archive()
            print("The best cost in iteration " + str(i) +
                  " is: " + str(iteration_best_cost))
            if index < len(select_good_costs_set):
                print("The best cost is given by select_good_params_set")
            else:
                print("The best cost is given by select_random_params_set")
            # print(self.window_params_set)
            print(self.window_costs_set)
            weights = self.net.get_weights()
            for j in range(len(weights)):
                weights[j] = weights[j].flatten()
                self.last_neural_net_weight[j] = self.last_neural_net_weight[j].flatten()
            delta_weights = abs(np.hstack(tuple(weights)) - np.hstack(tuple(self.last_neural_net_weight)))
            self.last_neural_net_weight = self.net.get_weights()
            print(np.sum(delta_weights))

        print("The best parameters: " + str(self.best_params))
        print("The best cost: " + str(self.best_cost))
        self._save_archive()

    def _save_archive(self):
        save_params_set = None
        # K-聚类获得典型参数
        self.k_means.fit(self.train_params_set)
        labels = self.k_means.predict(self.train_params_set)
        for i in range(self.save_params_set_size):
            params_subset = self.train_params_set[labels == i]
            costs_subset = self.train_costs_set[labels == i]
            index = np.argmin(costs_subset)
            if save_params_set is None:
                save_params_set = np.array([params_subset[index]])
            else:
                save_params_set = np.vstack(
                    (save_params_set, params_subset[index]))
        self.archive.update({'save_params_set': save_params_set})
        self.archive.update({'neural_net_archive_filename': self.net.save()})
        utilities.save_dict_to_txt_file(self.archive, self.archive_filename)

    def plot_best_costs_list(self):
        x_axis = np.arange(start=0, stop=len(
            self.best_costs_list), step=1, dtype=int)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.plot(x_axis, self.best_costs_list)
        plt.show()
