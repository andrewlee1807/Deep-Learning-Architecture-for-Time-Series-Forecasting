#  Copyright (c) 2022 Andrew
#  Email: andrewlee1807@gmail.com

from sklearn.model_selection import train_test_split
import numpy as np


def pattern(sequence: np.array, kernel_size: int, gap=7):
    head_kernel_size = tail_kernel_size = kernel_size // 2

    # Padding
    padding = np.zeros(gap + tail_kernel_size)
    new_sequence = np.concatenate([sequence, padding])

    def generate_index(ix):
        for i in range(0, head_kernel_size):  # gen index from head
            list_ix.append(ix + i)
        for j in range(0, tail_kernel_size):  # gen index from tail
            list_ix.append(ix + gap + j)

    list_ix = []
    # ix_padding = len(sequence) - (gap + tail_kernel_size)
    # align sequence
    for node_index in range(0, len(sequence)):
        generate_index(node_index)
    new_sequence = new_sequence[list_ix]

    return new_sequence


class TimeSeriesGenerator:
    """
    This class only support to prepare training (backup to TSF class)
    """

    def __init__(
            self,
            data,
            input_width: int,
            output_width: int,
            shift=1,
            batch_size=32,
            train_ratio=None,
            shuffle=False,
    ):
        """
        return:
        data_train,
        data_valid,
        data_test,
        function: inverse_scale_transform
        """
        self.data_train = None
        self.data_test = None
        self.scaler_x = None
        self.scaler_y = None
        self.raw_data = data
        self.input_width = input_width
        self.output_width = output_width
        self.shift = shift
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.split_data(train_ratio)
        self.data_train = self.build_tsd(self.X_train)
        self.data_valid = self.build_tsd(self.X_valid)
        if self.X_test is not None:
            self.data_test = self.build_tsd(self.X_test)
        else:
            self.data_test = None

        # self.normalize_data()

    def split_data(self, train_ratio):
        self.X_test = None  # No testing, using whole data to train
        X_train = self.raw_data
        if train_ratio is not None:
            X_train, self.X_test = train_test_split(
                self.raw_data, train_size=train_ratio, shuffle=self.shuffle
            )
        self.X_train, self.X_valid = train_test_split(
            X_train, train_size=0.9, shuffle=self.shuffle
        )

    def inverse_scale_transform(self, y_predicted):
        """
        un-scale predicted output
        """
        if self.scaler_y is not None:
            return self.scaler_y.inverse_transform(y_predicted)
        return y_predicted

    def re_arrange_sequence(self):
        # self.data_train =
        pass

    def normalize_data(self, standardization_type=1):
        """The mean and standard deviation should only be computed using the training data so that the models
        have no access to the values in the validation and test sets.
        1: MinMaxScaler, 2: StandardScaler, 3: RobustScaler, 4: PowerTransformer
        """
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import PowerTransformer
        from sklearn.preprocessing import RobustScaler
        from sklearn.preprocessing import StandardScaler

        standardization_methods = {
            1: MinMaxScaler,
            2: StandardScaler,
            3: RobustScaler,
            4: PowerTransformer,
        }
        standardization_method = standardization_methods[standardization_type]
        scaler_x = standardization_method()
        scaler_x.fit(self.data_train[0])
        scaler_y = standardization_method()
        scaler_y.fit(self.data_train[1])
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

        self.data_train = (
            scaler_x.transform(self.data_train[0]),
            scaler_y.transform(self.data_train[1]),
        )
        # converting into L.S.T.M format
        self.data_train = self.data_train[0][..., np.newaxis], self.data_train[1]
        self.data_valid = (
            scaler_x.transform(self.data_valid[0]),
            scaler_y.transform(self.data_valid[1]),
        )
        self.data_valid = self.data_valid[0][..., np.newaxis], self.data_valid[1]
        if self.data_test is not None:
            self.data_test = (
                scaler_x.transform(self.data_test[0]),
                scaler_y.transform(self.data_test[1]),
            )
            self.data_test = self.data_test[0][..., np.newaxis], self.data_test[1]

    def build_tsd(self, data):
        X_data, y_label = [], []
        if self.input_width >= len(data) - self.output_width - 168:
            raise ValueError(
                f"Cannot devide sequence with length={len(data)}. The dataset is too small to be used input_length= {self.input_width}. Please reduce your input_length"
            )

        for i in range(self.input_width, len(data) - self.output_width):
            X_data.append(data[i - self.input_width: i])
            y_label.append(data[i: i + self.output_width])

        X_data, y_label = np.array(X_data), np.array(y_label)

        return X_data, y_label


from utils.datasets import *


def get_all_data_supported():
    return list(CONFIG_PATH.keys())


class Dataset:
    """
    Dataset class hold all the dataset via dataset name
    :function:
    - Load dataset
    """

    def __init__(self, dataset_name):
        dataset_name = dataset_name.upper()
        if dataset_name not in get_all_data_supported():
            raise f"Dataset name {dataset_name} isn't supported"
        self.dataset_name = dataset_name
        # DataLoader
        self.dataloader = self.__load_data()

    def __load_data(self):
        if self.dataset_name == cnu_str:
            return CNU()
        elif self.dataset_name == comed_str:
            return COMED()
