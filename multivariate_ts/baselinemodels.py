#  Copyright (c) 2023 Andrew
#  Email: andrewlee1807@gmail.com

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.losses import Huber


class Baseline:
    def __init__(self):
        self.model = None

    def initialize_model(self, model_class, input_width, num_hidden_layer, num_features, output_length):
        self.model = Sequential()
        self.model.add(model_class(num_hidden_layer[0], input_shape=(input_width, num_features), return_sequences=True))
        for i in range(1, len(num_hidden_layer)):
            self.model.add(model_class(num_hidden_layer[i], return_sequences=True))
        self.model.add(Dense(output_length))

    def compile_model(self, optimizer, metrics):
        self.model.compile(optimizer=optimizer,
                           loss=Huber(),
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
        self.model = Sequential()
        self.model.add(Dense(num_hidden_layer[0], input_shape=(input_width, num_features), activation='relu'))
        for i in range(1, len(num_hidden_layer)):
            self.model.add(Dense(num_hidden_layer[i], activation='sigmoid'))
        self.model.add(Dense(output_length))
