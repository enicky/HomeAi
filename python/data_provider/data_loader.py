import glob
import os
import re
import warnings
from utils.augmentation import run_augmentation_single

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import Dataset

from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')

import logging
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)



# noinspection DuplicatedCode
class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True,
                 scaler='StandardScaler', timeenc=0, freq='h', lag=0, seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        if scale:
            if scaler == 'StandardScaler':
                self.scaler = StandardScaler()  # normalize mean to 0, variance to 1
            elif scaler == 'MinMaxScaler':
                self.scaler = MinMaxScaler()  # normalize to [0, 1]
            elif scaler == 'MaxAbsScaler':
                self.scaler = MaxAbsScaler()  # normalize to [-1, 1]
            elif scaler == 'BoxCox':
                self.scaler = PowerTransformer(method='yeo-johnson')  # box-cox transformation
            else:
                raise NotImplementedError
        else:
            self.scaler = None
        self.timeenc = timeenc
        self.freq = freq
        self.lag = lag

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("features should be 'M', 'MS' or 'S'!")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # add lag feature
        if self.lag > 0:
            for i in range(self.lag, 0, -1):
                label_data = data[:, -1]  # {ndarray: (N,)}
                lag_data = np.concatenate([np.zeros(i), label_data[:-i]], axis=0)  # {ndarray: (N,)}
                data = np.concatenate([lag_data.reshape(-1, 1), data], axis=1)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError('timeenc should be 0 or 1!')

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_all_data(self):
        return self.data_x, self.data_y, self.data_stamp


# noinspection DuplicatedCode
class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, features='S', data_path='ETTm1.csv', target='OT', scale=True,
                 scaler='StandardScaler', timeenc=0, freq='t', lag=0, seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        if scale:
            if scaler == 'StandardScaler':
                self.scaler = StandardScaler()  # normalize mean to 0, variance to 1
            elif scaler == 'MinMaxScaler':
                self.scaler = MinMaxScaler()  # normalize to [0, 1]
            elif scaler == 'MaxAbsScaler':
                self.scaler = MaxAbsScaler()  # normalize to [-1, 1]
            elif scaler == 'BoxCox':
                self.scaler = PowerTransformer(method='yeo-johnson')  # box-cox transformation
            else:
                raise NotImplementedError
        else:
            self.scaler = None
        self.timeenc = timeenc
        self.freq = freq
        self.lag = lag

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("features should be 'M', 'MS' or 'S'!")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # add lag feature
        if self.lag > 0:
            for i in range(self.lag, 0, -1):
                label_data = data[:, -1]  # {ndarray: (N,)}
                lag_data = np.concatenate([np.zeros(i), label_data[:-i]], axis=0)  # {ndarray: (N,)}
                data = np.concatenate([lag_data.reshape(-1, 1), data], axis=1)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError('timeenc should be 0 or 1!')

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_all_data(self):
        return self.data_x, self.data_y, self.data_stamp


# noinspection DuplicatedCode
class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True,
                 scaler='StandardScaler', timeenc=0, freq='h', lag=0, seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        logPrefix = '[Dataset_Custom:init]'
        # info
        logger.info(f'{logPrefix} size : {size}')
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_x = None
        self.data_y = None

        self.features = features
        self.target = target
        self.scale = scale
        logger.info(f'{logPrefix} Scale: {scale}')
        if scale:
            if scaler == 'StandardScaler':
                self.scaler = StandardScaler()  # normalize mean to 0, variance to 1
            elif scaler == 'MinMaxScaler':
                self.scaler = MinMaxScaler()  # normalize to [0, 1]
            elif scaler == 'MaxAbsScaler':
                self.scaler = MaxAbsScaler()  # normalize to [-1, 1]
            elif scaler == 'BoxCox':
                self.scaler = PowerTransformer(method='yeo-johnson')  # box-cox transformation
            else:
                raise NotImplementedError
        else:
            self.scaler = None
        self.timeenc = timeenc
        self.freq = freq
        self.lag = lag

        self.path = os.path.join(root_path, data_path)
        logger.info(f'{logPrefix} start reading data from {self.path}')
        self.__read_data__()

    def __read_data__(self):
        logPrefix = '[Dataset_Custom:__read_data__]'
        # read raw data
        print(f'[Dataset_Custom:__read_data__] start reading from {os.getcwd()}/{self.path}')
        df_raw = pd.read_csv(self.path)
        logger.debug(f'{logPrefix} read data : ')
        logger.debug(df_raw.head())
                    
        #print(f'[] result reading : {df_raw}')

        # check if exist nan
        if df_raw.isnull().values.any():
        #    print(f'[] There are some missing values ...')
            df_raw = interpolate_missing(df_raw)

        # df_raw.columns: ['date', ...(other features), target feature]
        cols = list(df_raw.columns)
        #print(f'[] cols : {cols}')
        if 'date' in cols:
            date_column = 'date'
        else:
            raise NotImplementedError("Make sure your datasets contain the column named 'date'!")

        # remove date and target columns
        #print(f'[] remove target {self.target}')
        cols.remove(self.target)
        cols.remove(date_column)
        if "Time" in cols:
            cols.remove('Time')
        
        # reorganize df_raw
        df_raw = df_raw[[date_column] + cols + [self.target]]
        logger.debug(f'{logPrefix} After processing result in df_raw is:')
        logger.debug(df_raw.head())

        # divide data into train, vali, test parts
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        logger.debug(f'{logPrefix} num_train {num_train} - num_test {num_test} - num_vali {num_vali}')

        # get boarders of the data
        # set_type: {'train': 0, 'val': 1, 'test': 2}
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # select features
        if self.features == 'M' or self.features == 'MS':
            # select features except the first
            cols_data = df_raw.columns[1:]
            logger.info(f'{logPrefix} cols_data = {cols_data}')
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # select features only the target
            df_data = df_raw[[self.target]]
        else:
            raise NotImplementedError

        # apply standard scaler if needed
        if self.scale:
            #print(f'[] standard scaler ... ')
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        logger.info(f'{logPrefix} Do we need to use lag ? {self.lag}')

        # add lag feature
        if self.lag > 0:
            for i in range(self.lag, 0, -1):
                label_data = data[:, -1]  # {ndarray: (N,)}
                lag_data = np.concatenate([np.zeros(i), label_data[:-i]], axis=0)  # {ndarray: (N,)}
                data = np.concatenate([lag_data.reshape(-1, 1), data], axis=1)

        # extract date column
        df_stamp = df_raw[[date_column]][border1:border2]
        df_stamp[date_column] = pd.to_datetime(df_stamp.date)

        logger.debug(f'{logPrefix} timeenc : {self.timeenc}')
        # encode time
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([date_column], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[date_column].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError('timeenc should be 0 or 1!')

        
        # output data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_scaler(self):
        return self.scaler

    def get_all_data(self):
        return self.data_x, self.data_y, self.data_stamp

    @staticmethod
    def _visual(corr, name):
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.subplots(figsize=(16, 16))
        plt.rc('font', family='Times New Roman', size=16)

        sns.heatmap(corr, center=0, annot=True, vmax=1.0, vmin=-1, square=True, cmap="Blues")
        plt.title("Heatmap", fontsize=18)
        # plt.savefig(f'{name}.pdf', bbox_inches='tight', format='pdf')
        plt.savefig(f'{name}.png', bbox_inches='tight')
        print(f'[] save to {name}.png')
        plt.show()

    def get_new_indexes(self, tolerance, visual=False):
        # get data except the last column
        data = self.data_x[:, :-1]

        # get correlation matrix
        corr_data = pd.DataFrame(data)
        corr = corr_data.corr()

        # visual if needed
        if visual:
            self._visual(corr, 'corr1')

        # traverse the upper triangle of the correlation matrix
        # rank correlation coefficient
        ranked_corr_data = []  # a list of tuples, like [((2,3), 0.95), ...], (2,3) is the index, 0.95 is the value
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                ranked_corr_data.append(((i, j), np.abs(corr.iloc[i, j])))
        ranked_corr_data.sort(key=lambda x: x[1], reverse=True)

        # group those features with high correlation
        new_indexes = []
        between_group = False
        groups = []
        groups_len = -1
        grouped_num = 0
        between_groups_links = []
        between_groups_link_num = 0
        total_num = corr.shape[0]
        for item_index in range(len(ranked_corr_data)):
            item = ranked_corr_data[item_index]
            i = item[0][0]
            j = item[0][1]
            value = item[1]
            if not between_group:
                # start to group within groups
                if value <= tolerance:
                    # check tolerance
                    between_group = True
                    if grouped_num < total_num:
                        # fill the rest
                        for k in range(total_num):
                            find = False
                            for group in groups:
                                if k in group:
                                    find = True
                                    break
                            if not find:
                                groups.append([k])
                                grouped_num += 1
                else:
                    # scan
                    i_group_index = []
                    j_group_index = []
                    for group_index in range(len(groups)):
                        group = groups[group_index]
                        if i in group:
                            i_group_index.append(group_index)
                        if j in group:
                            j_group_index.append(group_index)
                    # add
                    if len(i_group_index) == 0 and len(j_group_index) == 0:
                        groups.append([i, j])
                        grouped_num += 2
                    elif len(i_group_index) == 1 and len(j_group_index) == 0:
                        groups[i_group_index[0]].append(j)
                        grouped_num += 1
                    elif len(i_group_index) == 0 and len(j_group_index) == 1:
                        groups[j_group_index[0]].append(i)
                        grouped_num += 1
                if grouped_num >= total_num:
                    between_group = True
            if between_group:
                if groups_len == -1:
                    groups_len = len(groups)

                # start to group between groups
                # find link
                link = []
                for k in range(groups_len):
                    group = groups[k]
                    if i in group or j in group:
                        link.append(k)

                # link between groups
                if len(link) == 2:
                    link_1 = link[0]
                    link_2 = link[1]

                    # check if the link is inside the between_groups_link
                    flag = False
                    in_num = 0
                    link_1_index = -1
                    link_2_index = -1
                    for between_groups_link_index in range(len(between_groups_links)):
                        between_groups_link = between_groups_links[between_groups_link_index]
                        if link_1 in between_groups_link and link_2 in between_groups_link:
                            flag = True
                        elif link_1 in between_groups_link:
                            in_num += 1
                            link_1_index = between_groups_link_index
                        elif link_2 in between_groups_link:
                            in_num += 1
                            link_2_index = between_groups_link_index
                    if flag:
                        continue

                    if in_num == 1:
                        # link to exist between_groups_link
                        if link_1_index != -1:
                            between_groups_link = between_groups_links[link_1_index]
                            if link_1 in between_groups_link:
                                if link_1 == between_groups_link[0]:
                                    between_groups_link.insert(0, link_2)
                                    between_groups_link_num += 1
                                elif link_1 == between_groups_link[-1]:
                                    between_groups_link.append(link_2)
                                    between_groups_link_num += 1
                        else:
                            between_groups_link = between_groups_links[link_2_index]
                            if link_2 in between_groups_link:
                                if link_2 == between_groups_link[0]:
                                    between_groups_link.insert(0, link_1)
                                    between_groups_link_num += 1
                                elif link_2 == between_groups_link[-1]:
                                    between_groups_link.append(link_1)
                                    between_groups_link_num += 1
                    elif in_num == 2:
                        # link two between_groups_link
                        between_groups_link_1 = between_groups_links[link_1_index]
                        between_groups_link_2 = between_groups_links[link_2_index]
                        if link_1 == between_groups_link_1[0] and link_2 == between_groups_link_2[0]:
                            for element in between_groups_link_2:
                                between_groups_link_1.insert(0, element)
                            between_groups_links.remove(between_groups_link_2)
                        elif link_1 == between_groups_link_1[0] and link_2 == between_groups_link_2[-1]:
                            between_groups_link_2.extend(between_groups_link_1)
                            between_groups_links.remove(between_groups_link_1)
                        elif link_1 == between_groups_link_1[-1] and link_2 == between_groups_link_2[0]:
                            between_groups_link_1.extend(between_groups_link_2)
                            between_groups_links.remove(between_groups_link_2)
                        elif link_1 == between_groups_link_1[-1] and link_2 == between_groups_link_2[-1]:
                            between_groups_link_2.reverse()
                            for element in between_groups_link_2:
                                between_groups_link_1.append(element)
                            between_groups_links.remove(between_groups_link_2)
                    elif in_num == 0:
                        # add new between_groups_link
                        between_groups_links.append([link_1, link_2])
                        between_groups_link_num += 2

                if between_groups_link_num >= groups_len and len(between_groups_links) == 1:
                    # start to adjust the sequence of groups
                    new_indexes = []
                    for i in between_groups_links[0]:
                        new_indexes.extend(groups[i])
                    new_indexes.append(self.data_x.shape[1] - 1)
                    break

        # visual if needed
        if visual:
            data = self.data_x[:, new_indexes]
            data = data[:, :-1]
            corr_data = pd.DataFrame(data)
            corr = corr_data.corr()
            self._visual(corr, 'corr2(1)')
            new_indexes.reverse()
            data = self.data_x[:, new_indexes]
            data = data[:, :-1]
            corr_data = pd.DataFrame(data)
            corr = corr_data.corr()
            self._visual(corr, 'corr2(2)')
            new_indexes.reverse()
        return new_indexes

    def set_new_indexes(self, new_indexes):
        self.data_x = self.data_x[:, new_indexes]
        self.data_y = self.data_y[:, new_indexes]


# noinspection DuplicatedCode
class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min', lag=0,
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        #print(f'[] Current flag : {self.flag} -> {self.root_path}')
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        #print(f'[] dataset groups : {dataset.groups}')
        datasetvalues = dataset.values[dataset.groups == self.seasonal_patterns]

        #print(f'[] datasetvalues = {datasetvalues}')
        training_values = [v[~np.isnan(v)] for v in datasetvalues]  # split different frequencies
        #print(f'[] training_values = {training_values}')
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


# noinspection DuplicatedCode
class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.args = args
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# noinspection DuplicatedCode
class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.args = args
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# noinspection DuplicatedCode
class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.args = args
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# noinspection DuplicatedCode
class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.args = args
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# noinspection DuplicatedCode
class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.args = args
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


# noinspection DuplicatedCode
class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.max_seq_len = None
        self.class_names = None
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
            flag: flags.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern = '*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                   replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
