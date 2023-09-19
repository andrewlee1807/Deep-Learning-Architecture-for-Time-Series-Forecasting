#  Copyright (c) 2023 Andrew
#  Email: andrewlee1807@gmail.com
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import sys
from abc import ABC

sys.path.insert(0, '../')

import os
import pandas as pd
import argparse
from matplotlib import pyplot as plt

from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from utils.logging import arg_parse, warming_up, close_logging
from utils.data import Dataset, TimeSeriesGenerator

import keras_tuner as kt


class StrideLayer(layers.Layer):
    def __init__(self,
                 nb_stride=3,
                 nb_filters=64,
                 kernel_size=3,
                 padding='causal',
                 dropout_rate=0.0,
                 init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                 name="DilatedLayer", **kwargs):
        super(StrideLayer, self).__init__(name=name, **kwargs)

        self.conv1 = layers.Conv1D(filters=nb_filters,
                                   kernel_size=kernel_size,
                                   strides=nb_stride,
                                   padding=padding,
                                   name='conv1D',
                                   kernel_initializer=init)

        self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation('relu')
        self.drop1 = layers.Dropout(rate=dropout_rate)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x)
        return x


class StrideDilatedNet(tf.keras.Model):
    def __init__(self,
                 list_stride=(3, 3),
                 nb_filters=64,
                 kernel_size=3,
                 padding='causal',
                 target_size=24,
                 dropout_rate=0.0,
                 **kwargs):
        self.nb_filters = nb_filters
        self.list_stride = list_stride
        self.kernel_size = kernel_size
        self.padding = padding

        super(StrideDilatedNet, self).__init__(**kwargs)
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        self.stride_blocks = []

        for i, d in enumerate(self.list_stride):
            stride_block_filters = self.nb_filters
            self.stride_blocks.append(
                StrideLayer(nb_stride=self.list_stride[i],
                            nb_filters=stride_block_filters,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            init=init,
                            dropout_rate=dropout_rate,
                            name=f"DilatedLayer_{i}")
            )

        self.slicer_layer = layers.Lambda(lambda tt: tt[:, -1, :], name='Slice_Output')

        self.dense = layers.Dense(units=target_size)

    def call(self, inputs, training=True):
        x = inputs
        for stride_block in self.stride_blocks:
            x = stride_block(x)
        x = self.slicer_layer(x)
        x = self.dense(x)
        return x


class HOModel(kt.HyperModel, ABC):
    def __init__(self, input_width, output_width, num_features):
        self.output_width = output_width
        self.input_width = input_width
        self.num_features = num_features

    def build(self, hp):
        kernel_size = hp.Choice('kernel_size', values=[2, 3, 5, 7])
        nb_filters = hp.Choice('nb_filters', values=[8, 16, 32, 64])
        dropout_rate = hp.Float('dropout_rate', 0, 0.5, step=0.1, default=0.5)
        layer_stride1 = hp.Choice('layer_stride1', values=range(1, 24))
        layer_stride2 = hp.Choice('layer_stride2', values=range(1, 7))

        model = StrideDilatedNet(list_stride=(layer_stride1, layer_stride2),
                                 nb_filters=nb_filters,
                                 kernel_size=kernel_size,
                                 padding='causal',
                                 target_size=self.output_width,
                                 dropout_rate=dropout_rate)

        # print model
        input_test = Input(shape=(self.input_width, self.num_features))
        model(input_test)
        model.summary()

        model.compile(loss=Huber(),
                      optimizer='adam',
                      metrics=['mse', 'mae'])

        return model


def auto_training(data_seq, config):
    input_width = config['input_width']
    num_features = len(config['features'])
    max_trials = config['max_trials']

    print("Building time series generator...")
    tsf = TimeSeriesGenerator(data=data_seq,
                              config=config,
                              normalize_type=1,
                              shuffle=False)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   mode='min')

    reduceLR = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)

    callbacks = [
        early_stopping,
        reduceLR
    ]

    # Search model
    exp_path = f'{config["output_dir"]}/{config["dataset_name"]}_stride_Tune/Bayesian/{config["output_length"]}'
    tuning_path = exp_path + "/models"

    if os.path.isdir(tuning_path):
        import shutil
        shutil.rmtree(tuning_path)

    model_builder = HOModel(input_width=input_width,
                            output_width=config['output_length'],
                            num_features=num_features)

    tuner = kt.BayesianOptimization(
        model_builder,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=max_trials,
        seed=42,
        directory=tuning_path)

    print("Searching hyperparameters...")

    tuner.search(tsf.data_train[0], tsf.data_train[1],
                 validation_data=tsf.data_valid,
                 callbacks=[TensorBoard(exp_path + "/log")],
                 epochs=10)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model_best = tuner.hypermodel.build(best_hps)

    # Train real model_searching
    print(f"""
        kernel_size {best_hps.get('kernel_size')},  and
        nb_filters: {best_hps.get('nb_filters')}, 
        dropout_rate: {best_hps.get('dropout_rate')}
        layer_stride1: {best_hps.get('layer_stride1')}
        layer_stride2: {best_hps.get('layer_stride2')}
        """)

    print('Train...')

    history = model_best.fit(x=tsf.data_train[0],
                             y=tsf.data_train[1],
                             validation_data=tsf.data_valid,
                             epochs=config["epochs"],
                             callbacks=[callbacks],
                             verbose=2,
                             use_multiprocessing=True)

    print("=============================================================")
    print("Minimum val mse:")
    print(min(history.history['val_mse']))
    print("Minimum training mse:")
    print(min(history.history['mse']))
    result = model_best.evaluate(tsf.data_test[0], tsf.data_test[1], batch_size=1,
                                 verbose=2,
                                 use_multiprocessing=True)

    print("Evaluation result: ")
    print(config['output_length'], result[1], result[2])

    result_file = f'{os.path.join(config["output_dir"], config["dataset_name"])}_evaluation_result.txt'
    file = open(result_file, 'a')
    file.write(f'{config["output_length"]},{result[1]},{result[2]}\n')
    file.close()


def get_dataset(dataset_name, features):
    # Load dataset
    dataset = Dataset(dataset_name=dataset_name)
    data = dataset.dataloader.export_the_sequence(features)
    return data


def main():
    args = arg_parse(argparse.ArgumentParser())
    config = warming_up(args)
    # Settings:
    dataset_name = args.dataset_name
    features = config['features']

    data_seq = get_dataset(dataset_name, features)

    auto_training(data_seq, config)

    if args.write_log_file:
        close_logging(config["file"], config["orig_stdout"])


if __name__ == '__main__':
    main()
