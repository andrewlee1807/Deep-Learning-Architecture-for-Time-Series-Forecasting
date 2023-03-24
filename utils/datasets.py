#  Copyright (c) 2022 Andrew
#  Email: andrewlee1807@gmail.com
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# List dataset name
cnu_str = "CNU"
comed_str = "COMED"
spain_str = "SPAIN"
household_str = "HOUSEHOLD"
gyeonggi_str = "GYEONGGI"

# Dataset path
CONFIG_PATH = {
    # cnu_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/%EA%B3%B5%EB%8C%807%ED%98%B8%EA%B4%80_HV_02.csv",
    cnu_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/%EA%B3%B5%EB%8C%807%ED%98%B8%EA%B4%80_HV_02_datetime.csv",
    comed_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/COMED_hourly.csv",
    spain_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/spain/spain_ec_499.csv",
    household_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/household_daily_power_consumption.csv",
    gyeonggi_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/gyeonggi_univariable/2955_1hour.csv",
    # gyeonggi_str: "https://raw.githubusercontent.com/andrewlee1807/Weights/main/datasets/gyeonggi_multivariable/2955_1hour.csv"
}


class DataLoader(ABC):
    """
    Class to be inheritance from others dataset
    """

    def __init__(self, path_file, data_name):
        self.raw_data = None
        if path_file is None:
            self.path_file = CONFIG_PATH[data_name]
        else:
            self.path_file = path_file
        print("Reading data from: {}".format(self.path_file))

    def read_data_frame(self):
        return pd.read_csv(self.path_file)

    def read_a_single_sequence(self):
        return np.loadtxt(self.path_file)

    @abstractmethod
    def export_a_single_sequence(self):
        pass

    @abstractmethod
    def export_the_sequence(self, feature_names):
        pass


# GYEONGGI dataset
class GYEONGGI(DataLoader):
    def __init__(self, path_file=None):
        super(GYEONGGI, self).__init__(path_file, gyeonggi_str)
        self.raw_data = self.read_data_frame()

    def read_data_frame(self):
        return pd.read_csv(self.path_file, header=0, sep='\t')

    def export_a_single_sequence(self):
        return self.raw_data['Amount of Consumption'].to_numpy()  # a single sequence

    def export_the_sequence(self, feature_names):
        return self.raw_data[feature_names].to_numpy()


# CNU dataset
class CNU(DataLoader):
    def __init__(self, path_file=None):
        super(CNU, self).__init__(path_file, cnu_str)
        self.raw_data = self.read_data_frame()

    # def export_a_single_sequence(self):
    #     return self.raw_data  # a single sequence

    def read_data_frame(self):
        return pd.read_csv(self.path_file, header=0, sep=',')

    def export_a_single_sequence(self):
        return self.raw_data['전력사용량'].to_numpy()  # a single sequence

    def export_the_sequence(self, feature_names):
        return self.raw_data[feature_names].to_numpy()


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

    def export_a_single_sequence(self):
        # Pick the customer no 20
        return self.dataframe.loc[:, 20]  # a single sequence


def fill_missing(data):
    one_day = 23 * 60
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if np.isnan(data[row, col]):
                data[row, col] = data[row - one_day, col]


class HouseholdDataLoader:
    def __init__(self, data_path=None):
        self.df = None
        self.data_by_days = None
        self.data_by_hour = None
        self.data_path = data_path
        self.load_data()

    def load_data(self):

        df = pd.read_csv(self.data_path, sep=';',
                         parse_dates={'dt': ['Date', 'Time']},
                         infer_datetime_format=True,
                         low_memory=False, na_values=['nan', '?'],
                         index_col='dt')

        droping_list_all = []
        for j in range(0, 7):
            if not df.iloc[:, j].notnull().all():
                droping_list_all.append(j)
        for j in range(0, 7):
            df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].mean())

        fill_missing(df.values)

        self.df = df
        self.data_by_days = df.resample('D').sum()  # all the units of particular day
        self.data_by_hour = df.resample('H').sum()  # all the units of particular day
