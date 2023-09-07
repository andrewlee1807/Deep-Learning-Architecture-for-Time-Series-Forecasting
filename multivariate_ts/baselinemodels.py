#  Copyright (c) 2023 Andrew
#  Email: andrewlee1807@gmail.com

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Flatten, Input
from tensorflow.keras.losses import Huber, MeanSquaredError


class Baseline:
    def __init__(self):
        self.model = None

    def initialize_model(self, model_class, input_width, num_hidden_layer, num_features, output_length):
        self.model = Sequential()
        self.model.add(model_class(num_hidden_layer[0], input_shape=(input_width, num_features), return_sequences=True))
        for i in range(1, len(num_hidden_layer) - 1):
            self.model.add(model_class(num_hidden_layer[i], return_sequences=True))
        self.model.add(model_class(num_hidden_layer[-1], return_sequences=False))
        self.model.add(Dense(output_length))

    def compile_model(self, optimizer, metrics):
        self.model.compile(optimizer=optimizer,
                           loss=MeanSquaredError(),
                           metrics=metrics)
        self.model.summary()


class LSTMModel(Baseline):
    def __init__(self, input_width, num_hidden_layer=(64, 32), num_features=1, output_length=1):
        model_class = LSTM
        super().__init__()
        self.initialize_model(model_class, input_width, num_hidden_layer, num_features, output_length)


class GRUModel(Baseline):
    def __init__(self, input_width, num_hidden_layer=(64, 32), num_features=1, output_length=1):
        model_class = GRU
        super().__init__()
        self.initialize_model(model_class, input_width, num_hidden_layer, num_features, output_length)


class MLPModel(Baseline):
    def __init__(self, input_width, num_hidden_layer=(100, 200), num_features=1, output_length=1):
        super().__init__()
        self.input_width = input_width
        self.num_features = num_features
        self.model = Sequential()
        # Flatten the input data, as MLPs require a 1D input
        self.model.add(Flatten(input_shape=(input_width, num_features)))
        for i in range(0, len(num_hidden_layer)):
            self.model.add(Dense(num_hidden_layer[i], activation='sigmoid'))
        self.model.add(Dense(output_length))

    def compile_model(self, optimizer, metrics):
        input_test = Input(shape=(self.input_width, self.num_features))
        self.model(input_test)
        self.model.summary()

        self.model.compile(optimizer=optimizer,
                           loss=Huber(),
                           metrics=metrics)
