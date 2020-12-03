import os
import datetime
import utilities
import numpy as np
from keras import models
from keras import layers
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf


class NeuralNet():

    def __init__(self,
                 min_boundary,
                 max_boundary,
                 costs_mean=0,
                 costs_stdev=1,
                 archive_dir=None,
                 start_datetime=None):
        self.params_mean = (max_boundary + min_boundary) / 2
        self.params_stdev = (max_boundary - min_boundary) / 2
        self.costs_mean = costs_mean
        self.costs_stdev = costs_stdev
        self.archive_dir = archive_dir
        self.start_datetime = start_datetime
        self.neural_net_file_prefix = 'neural_net_archive_'

    def init(self,
             num_params,
             layer_dims,
             train_threshold_ratio=0.01,
             batch_size=8,
             dropout_prob=0.5,
             regularisation_coefficient=1e-8):
        # 神经网络的超参数
        self.num_params = num_params
        self.layer_dims = layer_dims
        self.train_threshold_ratio = train_threshold_ratio
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.regularisation_coefficient = regularisation_coefficient

        # 添加 GELU 激活函数
        get_custom_objects().update(
            {'gelu': gelu})

        # 构造神经网络
        self.model = models.Sequential()
        prev_layer_dim = self.num_params
        for layer_dim in layer_dims:
            self.model.add(layers.Dense(layer_dim,
                                        activation='gelu',
                                        bias_initializer='glorot_uniform',
                                        kernel_regularizer=regularizers.l2(
                                            self.regularisation_coefficient),
                                        input_shape=(prev_layer_dim,)))
            self.model.add(layers.Dropout(self.dropout_prob))
            prev_layer_dim = layer_dim
        self.model.add(layers.Dense(1))
        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    def load(self, archive, filename):
        # 神经网络的超参数
        self.train_threshold_ratio = archive['train_threshold_ratio']
        self.batch_size = archive['batch_size']

        # 加载神经网络
        self.model = models.load_model(filename, custom_objects={'gelu': gelu})

    def save(self):
        filename = os.path.join(
            self.archive_dir, self.neural_net_file_prefix + self.start_datetime + '.h5')
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.model.save(filename)
        return filename

    def _scale_params(self, params_unscaled):
        params_scaled = params_unscaled - self.params_mean
        params_scaled /= self.params_stdev
        return params_scaled

    def _scale_costs(self, costs_unscaled):
        costs_scaled = costs_unscaled - self.costs_mean
        costs_scaled /= self.costs_stdev
        return costs_scaled

    def _unscale_params(self, params_scaled):
        params_unscaled = params_scaled * self.params_stdev
        params_unscaled += self.params_mean
        return params_unscaled

    def _unscale_cost(self, costs_scaled):
        costs_unscaled = costs_scaled * self.costs_stdev
        costs_unscaled += self.costs_mean
        return costs_unscaled

    def _loss_and_metrics(self, params, costs):
        loss, metrics = self.model.evaluate(params, costs, verbose=0)
        return loss, metrics

    def get_loss(self, params, costs):
        return self._loss_and_metrics(params, costs)[0]

    def fit(self, params, costs, max_epoch, validation_params=None, validation_costs=None):
        params_scaled = self._scale_params(params)
        costs_scaled = self._scale_costs(costs)
        if validation_params is None or validation_costs is None:
            early_stopping = EarlyStopping(
                monitor='loss', min_delta=self.train_threshold_ratio, patience=5000, mode='min')
            history = self.model.fit(params_scaled, costs_scaled, epochs=max_epoch,
                                    batch_size=self.batch_size, verbose=0, callbacks=[early_stopping])
        else:
            validation_params_scaled = self._scale_params(validation_params)
            validation_costs_scaled = self._scale_costs(validation_costs)
            early_stopping = EarlyStopping(
                monitor='val_loss', min_delta=self.train_threshold_ratio, patience=6, mode='min')

            history = self.model.fit(params_scaled, costs_scaled, epochs=max_epoch,
                                    batch_size=self.batch_size, verbose=0, callbacks=[early_stopping], validation_data=(validation_params_scaled, validation_costs_scaled))
        return history

    def predict_costs(self, params):
        costs_scaled = np.array(self.model.predict(params, verbose=0, use_multiprocessing=True)).flatten()
        costs_unscaled = self._unscale_cost(costs_scaled)
        return costs_unscaled

    def reset_weights(self):
        for layer in self.model.layers:
            for k, initializer in layer.__dict__.items():
                if "initializer" not in k:
                    continue
                var = getattr(layer, k.replace("_initializer", ""))
                var.assign(initializer(var.shape, var.dtype))

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
