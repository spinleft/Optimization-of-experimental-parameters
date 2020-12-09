import os
import numpy as np
import utilities
import multiprocessing


class Parameters():
    class Attrs():
        def __init__(self):
            self.cost = 0
            self.experiment_num = 0

        def is_new(self):
            return self.experiment_num == 0

        def get_cost(self):
            return self.cost

        def update(self, cost):
            self.cost = (self.experiment_num * self.cost +
                         cost) / (self.experiment_num + 1)
            self.experiment_num += 1

    def __init__(self, min_boundary, max_boundary, patch_length):
        self.min_boundary = np.array(min_boundary)
        self.max_boundary = np.array(max_boundary)
        if isinstance(patch_length, (float, int)):
            self.patch_length = np.array([patch_length] * len(min_boundary))
        else:
            self.patch_length = np.array(patch_length)
        self.grid = []
        for i in range(len(min_boundary)):
            self.grid.append(
                np.arange(min_boundary[i], max_boundary[i], self.patch_length[i]))
        self.indexes_range = np.array([len(self.grid[i])+1 for i in range(len(min_boundary))])
        self.root = dict()
        self.size = 0
        self.num_cores = os.cpu_count()
        # self.pool = multiprocessing.Pool(processes=self.num_cores)
        self.pool = None

    def __len__(self):
        return self.size

    def _get_indexes(self, params):
        indexes = []
        for i in range(len(params)):
            indexes.append(self.grid[i].searchsorted(
                params[i] - 0.5 * self.patch_length[i]))
        indexes = np.array(indexes)
        return indexes

    def _get_cost(self, indexes):
        target = self.root
        for index in indexes:
            if index not in target:
                return None
            target = target[index]
        return target.get_cost()

    def _insert(self, indexes, cost):
        curr_node = self.root
        for index in indexes[:-1]:
            curr_node = curr_node.setdefault(index, dict())
        attr = curr_node.setdefault(indexes[-1], self.Attrs())
        if attr.is_new():
            self.size += 1
        attr.update(cost)

    def _get_valid_uniform_params_set(self, params_set_size):
        rng = np.random.default_rng()
        params_set = np.zeros(shape=(params_set_size, len(self.min_boundary)))
        for i in range(params_set_size):
            while True:
                indexes = rng.integers(0, self.indexes_range)
                params_set[i] = self.min_boundary + indexes * self.patch_length
                if utilities.params_in_condition(params_set[i]):
                    break
        return params_set

    def _get_uniform_params_set(self, params_set_size):
        rng = np.random.default_rng()
        indexes_set = rng.integers(0, self.indexes_range, size=(
            params_set_size, len(self.min_boundary)))
        params_set = self.min_boundary + indexes_set * self.patch_length
        return params_set

    def _get_normal_params_set(self, base_indexes, stdev, params_set_size):
        rng = np.random.default_rng()
        stdev_scaled = stdev * self.indexes_range
        indexes_set = rng.normal(base_indexes, stdev_scaled, size=(
            params_set_size, len(self.min_boundary)))
        indexes_set = np.around(indexes_set)
        cond = indexes_set >= 0
        indexes_set = np.where(cond, indexes_set, 0)
        cond = indexes_set <= self.indexes_range
        indexes_set = np.where(cond, indexes_set, self.indexes_range)
        params_set = self.min_boundary + indexes_set * self.patch_length
        return params_set
    
    def _get_all_indexes(self, node):
        indexes = []
        for index in node:
            if isinstance(node[index], self.Attrs):
                return np.array(list(node.keys())).reshape(-1, 1)
            sub_indexes = self._get_all_indexes(node[index])
            indexes.append(np.concatenate(([[index]] * len(sub_indexes), sub_indexes), axis=1))
        return np.concatenate(indexes)
    
    def _get_all_costs(self, node):
        if isinstance(node, self.Attrs):
            return np.reshape(node.get_cost(), (1, ))
        costs = []
        for index in node:
            costs.append(self._get_all_costs(node[index]))
        return np.concatenate(costs)

    def insert(self, params, cost):
        indexes = self._get_indexes(params)
        self._insert(indexes, cost)

    def get_cost(self, params):
        indexes = self._get_indexes(params)
        return self._get_cost(indexes)

    def discretize(self, params):
        indexes = np.array(self._get_indexes(params))
        return self.min_boundary + indexes * self.patch_length

    def get_init_params_set(self, params_set_size):
        # 并行产生随机数
        if self.pool is not None:
            block_size = params_set_size // self.num_cores
            blocks = [block_size] * (self.num_cores - 1) + \
                [params_set_size - block_size * (self.num_cores - 1)]
            multiple_results = [self.pool.apply_async(self._get_valid_uniform_params_set, args=(
                sub_set_size)) for sub_set_size in blocks]
            params_set_list = [result.get() for result in multiple_results]
            params_set = np.concatenate(params_set_list)
        else:
            params_set = self._get_valid_uniform_params_set(params_set_size)
        return params_set

    def get_uniform_params_set(self, params_set_size):
        if self.pool is not None:
            block_size = params_set_size // self.num_cores
            blocks = [block_size] * (self.num_cores - 1) + \
                [params_set_size - block_size * (self.num_cores - 1)]
            multiple_results = [self.pool.apply_async(self._get_uniform_params_set, args=(
                sub_set_size)) for sub_set_size in blocks]
            params_set_list = [result.get() for result in multiple_results]
            params_set = np.concatenate(params_set_list)
        else:
            params_set = self._get_uniform_params_set(params_set_size)
        return params_set

    def get_normal_params_set(self, base_params, stdev, params_set_size):
        base_indexes = self._get_indexes(base_params)
        if self.pool is not None:
            block_size = params_set_size // self.num_cores
            blocks = [block_size] * (self.num_cores - 1) + \
                [params_set_size - block_size * (self.num_cores - 1)]
            multiple_results = [self.pool.apply_async(self._get_normal_params_set, args=(
                base_indexes, stdev, sub_set_size)) for sub_set_size in blocks]
            params_set_list = [result.get() for result in multiple_results]
            params_set = np.concatenate(params_set_list)
        else:
            params_set = self._get_normal_params_set(base_indexes, stdev, params_set_size)
        return params_set
    
    def get_all_params(self):
        all_indexes = self._get_all_indexes(self.root)
        return self.min_boundary + all_indexes * self.patch_length

    def get_all_costs(self):
        return self._get_all_costs(self.root)

if __name__ == '__main__':
    import time

    min_boundary = [0., 0., 0., 0., 0., 0., 0.]
    max_boundary = [1., 1., 1., 1., 1., 1., 1.]
    patch_length = 0.01
    params_set = Parameters(min_boundary, max_boundary, patch_length)
    start = time.time()
    for i in range(10):
        params = np.random.uniform(0, 1, size=(7,))
        params = params_set.discretize(params)
        print(params)
        cost = np.random.uniform(0, 1)
        params_set.insert(params, cost)
    all_params = params_set.get_all_params()
    print(all_params)
    end = time.time()
    print(end - start)
