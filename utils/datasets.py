#  Copyright (c) 2022 Andrew
#  Email: andrewlee1807@gmail.com
import numpy as np
import pandas as pd

# List dataset name
cnu_str = "CNU"
comed_str = "COMED"
spain_str = "SPAIN"

# Dataset path
CONFIG_PATH = {
    cnu_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/%EA%B3%B5%EB%8C%807%ED%98%B8%EA%B4%80_HV_02.csv",
    comed_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/COMED_hourly.csv",
    spain_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/spain/spain_ec_499.csv"
}


class DataLoader:
    """
    Class to be inheritance from others dataset
    """

    def __init__(self, path_file, data_name):
        self.raw_data = None
        if path_file is None:
            self.path_file = CONFIG_PATH[data_name]
        else:
            self.path_file = path_file

    def read_data_frame(self):
        return pd.read_csv(self.path_file)

    def read_a_single_sequence(self):
        return np.loadtxt(self.path_file)


# CNU dataset
class CNU(DataLoader):
    def __init__(self, path_file=None):
        super(CNU, self).__init__(path_file, cnu_str)
        self.raw_data = self.read_a_single_sequence()

    def export_sequences(self):
        return self.raw_data  # a single sequence


# COMED_hourly
class COMED(DataLoader):
    def __init__(self, path_file=None):
        super(COMED, self).__init__(path_file, comed_str)
        self.dataframe = self.read_data_frame()


# Spain dataset
class SPAIN(DataLoader):
    def __init__(self, path_file=None):
        super(SPAIN, self).__init__(path_file, spain_str)
        self.dataframe = self.read_data_frame()

    def export_sequences(self):
        # Pick the customer no 20
        return self.dataframe.loc[:, 20]  # a single sequence
