from delayedtcn.models import *
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

# List model name
model1_str = "Model1"
model2_str = "Model2"
model3_str = "Model3"


def build_callbacks(tensorboard_name=None):
    """Control and tracking learning process during training phase"""
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   mode='min')

    reduceLR = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)

    # TensorBoard(exp_path + "/log")

    callbacks = [
        early_stopping,
        reduceLR
    ]
    return callbacks


def initialize_model1(input_width, target_size, num_features):
    model = Model1(target_size=target_size)
    # print model
    input_test = Input(shape=(input_width, num_features))
    model(input_test)
    model.summary()
    # Build model
    model.compile(loss=Huber(),
                  optimizer='adam',
                  metrics=['mse', 'mae'])

    return model


def get_model(model_name: str, input_width: int, target_size: int, num_features=1) -> object:
    model_name = model_name.upper
    if model_name == model1_str:
        return initialize_model1(input_width, target_size, num_features)
    elif model_name == model2_str:
        pass
    elif model_name == model3_str:
        pass
    return None
