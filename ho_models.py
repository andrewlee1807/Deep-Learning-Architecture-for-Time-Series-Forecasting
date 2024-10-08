from delayedtcn.models import *
from tcnbased.tcn_family import TCN_Vanilla
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import keras_tuner as kt

# List model name
model1_str = "Model1"
model2_str = "Model2"
model3_str = "Model3"

# comparing model names
tcn_model_str = "TCN"


class HOModel(kt.HyperModel):
    def __init__(self, config, model_class):
        self.list_dilation = config['list_dilation'],
        self.output_length = config['output_length']
        self.num_features = config['num_features']
        self.input_width = config['input_width']
        self.model_class = model_class

    def build(self, hp):
        kernel_size = hp.Int('kernel_size',  # (kernel_size // 2) < layer_stride1:
                             min_value=12,
                             max_value=24,
                             step=2)
        # nb_filters = hp.Choice('nb_filters', values=[8, 16, 32, 64])

        layer_stride1 = hp.Int('layer_stride1',
                               min_value=kernel_size // 2,
                               max_value=24,
                               step=1)

        # layer_stride2 = hp.Choice('layer_stride2', values=range(1, 7))

        model = self.model_class(list_stride=list([layer_stride1, 1]),
                                 nb_filters=64,
                                 kernel_size=kernel_size,
                                 target_size=self.output_length)

        # print model
        input_test = Input(shape=(self.input_width, self.num_features))
        # model(input_test)
        model.summary(input_test)

        model.compile(loss=Huber(),
                      optimizer='adam',
                      metrics=['mse', 'mae'])

        return model


def build_callbacks(tensorboard_log_dir='logs', tensorboard_name=None):
    """Control and tracking learning process during training phase"""
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   mode='min')

    reduceLR = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)

    tensorboard_callback = TensorBoard(tensorboard_log_dir)

    callbacks = [
        # early_stopping,
        reduceLR,
        tensorboard_callback
    ]
    return callbacks


def initialize_model1(config):
    return HOModel(config=config, model_class=Model1)


def initialize_model2(config):
    return HOModel(config=config, model_class=Model2)


class HOTCNModel(kt.HyperModel):
    def __init__(self, config, model_class):
        self.output_length = config['output_length']
        self.num_features = config['num_features']
        self.input_width = config['input_width']
        self.model_class = model_class

    def build(self, hp):
        kernel_size = hp.Choice('kernel_size', values=[2, 3, 5, 7])
        nb_filters = hp.Choice('nb_filters', values=[16, 32, 64, 128])
        use_skip_connections = hp.Choice(
            'use_skip_connections', values=[True, False])

        use_batch_norm = hp.Choice(
            'use_batch_norm', values=[True, False])

        def temp(x): return 2 ** x

        def dilation_gen(x): return list(map(temp, range(x)))

        dilations = hp.Choice('dilations', values=list(range(2, 8)))

        model = TCN_Vanilla(input_width=self.input_width,
                            dilations=dilation_gen(dilations),
                            nb_filters=nb_filters,
                            kernel_size=kernel_size,
                            num_features=self.num_features,
                            use_skip_connections=use_skip_connections,
                            use_batch_norm=use_batch_norm,
                            target_size=self.output_length)

        input_test = Input(shape=(self.input_width, self.num_features))
        model.summary(input_test)

        model.compile(loss=Huber(),
                      optimizer='adam',
                      metrics=['mse', 'mae'])

        return model


def initialize_tcn_model(config):
    return HOTCNModel(config=config, model_class=TCN_Vanilla)


def get_model(model_name: str, config) -> object:
    model_name = model_name.upper()
    if model_name == model1_str.upper():
        return initialize_model1(config)
    elif model_name == model2_str.upper():
        return initialize_model2(config)
    elif model_name == model3_str.upper():
        pass
    elif model_name == tcn_model_str.upper():
        return initialize_tcn_model(config)

    return None
