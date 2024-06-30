import os
import time
import warnings

import numpy as np
import pandas
import torch

from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.m4_summary import M4Summary
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings('ignore')


# noinspection DuplicatedCode
class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, root_path, args, try_model=False, save_process=True):
        super(Exp_Short_Term_Forecast, self).__init__(root_path, args, try_model, save_process)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        return super()._build_model()

    def train(self, setting, check_folder=False, only_init=False):
        if check_folder:
            self._check_folders([self.root_checkpoints_path, self.root_process_path])

        checkpoints_path = os.path.join(self.root_checkpoints_path, setting)

        process_path = self.root_process_path + f'/{setting}/'
        if not os.path.exists(process_path) and not self.try_model:
            os.makedirs(process_path)
        self.process_file_path = process_path + f'{self.args.task_name}.txt'

        if only_init:
            return

        train_data, train_loader = self._get_data(data_flag='train', enter_flag='train', _try_model=self.try_model)
        vali_data, vali_loader = self._get_data(data_flag='val', enter_flag='train', _try_model=self.try_model)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.checkpoints_file, patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        stop_epochs = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # try model if needed
                if self.try_model:
                    # noinspection PyBroadException
                    try:
                        self.model(batch_x, None, dec_inp, None)
                        return True
                    except:
                        return False

                outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1,
                # :]))
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    _ = "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    self.print_content(_)
                    speed = (time.time() - time_now) / iter_count
                    # left time for all epochs
                    # left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # left time for current epoch
                    left_time = speed * (train_steps - i)
                    if left_time > 60 * 60:
                        _ = '\tspeed: {:.4f} s/iter; left time: {:.4f} hour'.format(speed, left_time / 60.0 / 60.0)
                    elif left_time > 60:
                        _ = '\tspeed: {:.4f} s/iter; left time: {:.4f} min'.format(speed, left_time / 60.0)
                    else:
                        _ = '\tspeed: {:.4f} s/iter; left time: {:.4f} second'.format(speed, left_time)
                    self.print_content(_)
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            current_epoch_time = time.time() - epoch_time
            if current_epoch_time > 60 * 60:
                _ = "Epoch: {}; cost time: {:.4f} hour".format(epoch + 1, current_epoch_time / 60.0 / 60.0)
            elif current_epoch_time > 60:
                _ = "Epoch: {}; cost time: {:.4f} min".format(epoch + 1, current_epoch_time / 60.0)
            else:
                _ = "Epoch: {}; cost time: {:.4f} second".format(epoch + 1, current_epoch_time)
            self.print_content(_)

            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            _ = ("Epoch: {0}, Steps: {1} --- Train Loss: {2:.7f}; Vali Loss: {3:.7f}; Test Loss: {4:.7f};".
                 format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            self.print_content(_)

            _ = early_stopping(vali_loss, self.model, checkpoints_path)
            if _ is not None:
                self.print_content(_)

            if early_stopping.early_stop:
                self.print_content("Early stopping")
                stop_epochs = epoch + 1
                break

            _ = adjust_learning_rate(model_optim, epoch + 1, self.args)
            if _ is not None:
                self.print_content(_)

        self.print_content("", True)

        best_model_path = checkpoints_path + '/' + self.checkpoints_file
        if os.path.exists(best_model_path):
            if self.device == torch.device('cpu'):
                self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
            else:
                self.model.load_state_dict(torch.load(best_model_path))

        if stop_epochs == 0:
            stop_epochs = self.args.train_epochs
        return stop_epochs

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]],
                                                                      None).detach().cpu()
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)

            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def test(self, setting, test=False, check_folder=False):
        _, train_loader = self._get_data(data_flag='train', enter_flag='test', _try_model=self.try_model)
        _, test_loader = self._get_data(data_flag='test', enter_flag='test', _try_model=self.try_model)
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:
            self.print_content('loading model')
            path = os.path.join(self.root_checkpoints_path, setting)
            best_model_path = path + '/' + self.checkpoints_file
            if os.path.exists(best_model_path):
                if self.device == torch.device('cpu'):
                    self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
                else:
                    self.model.load_state_dict(torch.load(best_model_path))
            else:
                raise FileNotFoundError('You need to train this model before testing it!')

        if check_folder:
            self._check_folders([self.root_test_results_path, self.root_m4_results_path])

        folder_path = self.root_test_results_path + f'/{setting}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None,
                                                                      dec_inp[id_list[i]:id_list[i + 1]], None)

                if id_list[i] % 1000 == 0:
                    self.print_content(f'id_list[i] = {id_list[i]}')

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        self.print_content('test shape:', preds.shape)
        self.print_content(f'preds : {preds}')

        # result save
        folder_path = self.root_m4_results_path + f'/{self.args.model}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')
        print(f'saving model {self.args.model} --> {folder_path + self.args.seasonal_patterns + "_forecast.csv"}')
        self.print_content(self.args.model)

        file_path = self.root_m4_results_path + f'/{self.args.model}/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, self.args.root_path)
            print(f'All forecasts are available ... start creating summary : {m4_summary}')
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            self.print_content('smape:', smape_results)
            self.print_content('mape:', mape)
            self.print_content('mase:', mase)
            self.print_content('owa:', owa_results)

            self.print_content("", True)

            return {
                'smape': smape_results
            }
        else:
            self.print_content('After all 6 tasks are finished, you can calculate the averaged index')

            return None
