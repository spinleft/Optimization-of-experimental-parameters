import os
import numpy as np
import utilities
import h5py


class Parameters():
    class Attrs():
        def __init__(self, uncer):
            self.cost = 0
            self.uncer = uncer
            self.experiment_num = 0

        def is_new(self):
            return self.experiment_num == 0

        def get_cost(self):
            return self.cost

        def get_biased_cost(self):
            return self.cost - self.uncer / np.sqrt(self.experiment_num)
        
        def get_experiment_num(self):
            return self.experiment_num
        
        def set_experiment_num(self, experiment_num):
            self.experiment_num = experiment_num

        def update(self, cost):
            self.cost = (self.experiment_num * self.cost +
                         cost) / (self.experiment_num + 1)
            self.experiment_num += 1

    def __init__(self, min_boundary, max_boundary, patch_length, uncer, archive_dir=None, start_datetime=None):
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
        self.indexes_range = np.array(
            [len(self.grid[i])+1 for i in range(len(min_boundary))])
        self.uncer = uncer

        self.root = dict()
        self.size = 0

        self.archive_dir = archive_dir
        self.start_datetime = start_datetime
        self.params_set_file_prefix = 'params_set_archive_'

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
    
    def _get_experiment_num(self, indexes):
        target = self.root
        for index in indexes:
            if index not in target:
                return None
            target = target[index]
        return target.get_experiment_num()
    
    def _set_experiment_num(self, indexes, experiment_num):
        target = self.root
        for index in indexes:
            if index not in target:
                return None
            target = target[index]
        return target.set_experiment_num(experiment_num)

    def _insert(self, indexes, cost):
        curr_node = self.root
        for index in indexes[:-1]:
            curr_node = curr_node.setdefault(index, dict())
        attr = curr_node.setdefault(indexes[-1], self.Attrs(self.uncer))
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
            indexes.append(np.concatenate(
                ([[index]] * len(sub_indexes), sub_indexes), axis=1))
        return np.concatenate(indexes)

    def _get_all_costs(self, node):
        if isinstance(node, self.Attrs):
            return np.reshape(node.get_cost(), (1, ))
        costs = []
        for index in node:
            costs.append(self._get_all_costs(node[index]))
        return np.concatenate(costs)

    def _get_all_biased_costs(self, node):
        if isinstance(node, self.Attrs):
            return np.reshape(node.get_biased_cost(), (1, ))
        costs = []
        for index in node:
            costs.append(self._get_all_costs(node[index]))
        return np.concatenate(costs)
    
    def _get_all_experiment_num(self, node):
        if isinstance(node, self.Attrs):
            return np.reshape(node.get_experiment_num(), (1, ))
        experiment_num = []
        for index in node:
            experiment_num.append(self._get_all_experiment_num(node[index]))
        return np.concatenate(experiment_num)

    def insert(self, params, cost):
        indexes = self._get_indexes(params)
        self._insert(indexes, cost)

    def get_cost(self, params):
        indexes = self._get_indexes(params)
        return self._get_cost(indexes)

    def get_experiment_num(self, params):
        indexes = self._get_indexes(params)
        return self._get_experiment_num(indexes)
    
    def set_experiment_num(self, params, experiment_num):
        indexes = self._get_indexes(params)
        self._set_experiment_num(indexes, experiment_num)

    def discretize(self, params):
        indexes = np.array(self._get_indexes(params))
        return self.min_boundary + indexes * self.patch_length

    def get_valid_params_set(self, params_set_size):
        return self._get_valid_uniform_params_set(params_set_size)

    def get_uniform_params_set(self, params_set_size):
        return self._get_uniform_params_set(params_set_size)

    def get_normal_params_set(self, base_params, stdev, params_set_size):
        base_indexes = self._get_indexes(base_params)
        return self._get_normal_params_set(base_indexes, stdev, params_set_size)

    def get_all_params(self):
        all_indexes = self._get_all_indexes(self.root)
        return self.min_boundary + all_indexes * self.patch_length

    def get_all_costs(self):
        return self._get_all_costs(self.root)

    def get_all_biased_costs(self):
        return self._get_all_biased_costs(self.root)

    def get_all_experiment_num(self):
        return self._get_all_experiment_num(self.root)
    
    def save(self):
        filename = os.path.join(
            self.archive_dir, self.params_set_file_prefix + self.start_datetime + '.h5')
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        all_params = self.get_all_params()
        all_costs = self.get_all_costs()
        all_experiment_num = self.get_all_experiment_num()
        f = h5py.File(filename, 'w')
        f.create_dataset('all_params', data=all_params)
        f.create_dataset('all_costs', data=all_costs)
        f.create_dataset('all_experiment_num', data=all_experiment_num)
        f.close()
        return filename

    def load(self, filename):
        f = h5py.File(filename, 'r')
        all_params = f['all_params'][()]
        all_costs = f['all_costs'][()]
        all_experiment_num = f['all_experiment_num'][()]
        for params, cost, experiment_num in zip(all_params, all_costs, all_experiment_num):
            self.insert(params, cost)
            self.set_experiment_num(params, experiment_num)


if __name__ == '__main__':
    import time

    min_boundary = [0., 0., 0., 0., 0., 0., 0.]
    max_boundary = [1., 1., 1., 1., 1., 1., 1.]
    patch_length = 0.01
    uncer = 0.1
    params_set = Parameters(min_boundary, max_boundary, patch_length, uncer)
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
