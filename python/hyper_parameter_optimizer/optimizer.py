import csv
import numpy as np
import os
import random
import time
import torch

from colorama import init, Fore
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_probability_forecasting import Exp_Probability_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from hyper_parameter_optimizer import basic_settings
from itertools import product
from tqdm import tqdm

from utils.tools import set_times_new_roman_font


from utils.print_args import print_args

import logging
logger = logging.getLogger(__name__)


class HyperParameterOptimizer(object):
    def __init__(self, script_mode, models=None, get_search_space=None, prepare_config=None, build_setting=None,
                 build_config_dict=None, set_args=None, get_fieldnames=None, get_model_id_tags=None,
                 check_jump_experiment=None, link_fieldnames_data=None, get_custom_test_time=None):
        # core settings
        self.script_mode = script_mode  # script mode

        # core functions
        # prepare config
        
        self.prepare_config = basic_settings.prepare_config if prepare_config is None else prepare_config
       
        # build setting, which is the unique identifier of the model
        self.build_setting = basic_settings.build_setting if build_setting is None else build_setting
        # build config dict - the data to be stored in files
        self.build_config_dict = basic_settings.build_config_dict if build_config_dict is None else build_config_dict
        # set args
        self.set_args = basic_settings.set_args if set_args is None else set_args
        # get tags
        self.get_model_id_tags = get_model_id_tags
        # get custom test time
        self.get_custom_test_time = get_custom_test_time

        # all mode settings
        self.seed = 2021  # random seed
        self.root_path = '.'  # root path of the data, script output, checkpoints, process and model results
        self.data_dir = 'data'  # data directory
        self.jump_csv_file = 'jump_data.csv'  # config data to be jumped
        self.data_csv_file = 'data.csv'  # default config data to be stored
        self.data_csv_file_format = 'data_{}.csv'  # config data to be stored in other processes
        self.scan_all_csv = False  # scan all config data in the path
        self.max_process_index = 0  # the maximum index of the processes
        self.save_process = True  # whether to save process
        self.add_tags = []  # added tags in the model id
        self.time_format = '%Y-%m-%d %H-%M-%S'  # time format in data and process
        self.diff_time_format = '%H:%M:%S'  # diff time format in data and process

        # init experiment and parameters
        self.Exp = None
        self._parameters = None
        self._task_names = None

        if not self.script_mode:
            # non-script mode settings
            # models
            self.models = models
            self.try_model = False  # whether to try model before running the experiments
            self.force_exp = False  # whether to force to run the experiments if already run them

            # fieldnames
            get_fieldnames = basic_settings.get_fieldnames if get_fieldnames is None else get_fieldnames
            self.all_fieldnames = get_fieldnames('all')  # all fieldnames
            self.checked_fieldnames = get_fieldnames('checked')  # checked fieldnames

            # search spaces
            self.get_search_space = get_search_space
            self.search_spaces = None
            self._check_required_fieldnames(get_fieldnames('required'))

            # non script mode functions
            self.check_jump_experiment = check_jump_experiment  # check if we need to jump the experiment
            self.link_fieldnames_data = link_fieldnames_data  # link data of fieldnames with other fieldnames

        get_fieldnames = basic_settings.get_fieldnames if get_fieldnames is None else get_fieldnames
        
        self.all_fieldnames = get_fieldnames('all')  # all fieldnames
        self.checked_fieldnames = get_fieldnames('checked')  # checked fieldnames
        # colorama init
        init(autoreset=True)

    def _check_required_fieldnames(self, fieldnames):
        for model in self.models:
            search_space = self._get_search_spaces()[model]
            # check if the required fieldnames are in the search space
            for fieldname in fieldnames:
                if fieldname not in search_space.keys():
                    raise ValueError(f'The required fieldname {fieldname} is not in the search space!')

    def config_optimizer_settings(self, root_path=None, data_dir=None, jump_csv_file=None, data_csv_file=None,
                                  data_csv_file_format=None, scan_all_csv=None, process_number=None,
                                  save_process=None, try_model=None, force_exp=None, add_tags=None):
        if root_path is not None:
            self.root_path = root_path
        if data_dir is not None:
            self.data_dir = data_dir
        if jump_csv_file is not None:
            self.jump_csv_file = jump_csv_file
        if data_csv_file is not None:
            self.data_csv_file = data_csv_file
        if data_csv_file_format is not None:
            self.data_csv_file_format = data_csv_file_format
        if scan_all_csv is not None:
            self.scan_all_csv = scan_all_csv
        if process_number is not None:
            self.max_process_index = process_number - 1
        if save_process is not None:
            self.save_process = save_process
        if not self.script_mode:
            if try_model is not None:
                self.try_model = try_model
            if force_exp is not None:
                self.force_exp = force_exp
            if add_tags is not None:
                self.add_tags = add_tags

    def get_optimizer_settings(self):
        core_setting = {
            'script_mode': self.script_mode,
        }

        all_mode_settings = {
            'random_seed': self.seed,
            'root_path': self.root_path,
            'data_dir': self.data_dir,
            'jump_csv_file': self.jump_csv_file,
            'data_csv_file': self.data_csv_file,
            'data_csv_file_format': self.data_csv_file_format,
            'scan_all_csv': self.scan_all_csv,
            'process_number': self.max_process_index + 1,
            'save_process': self.save_process,
        }

        non_script_mode_settings = {
            'models': self.models,
            'try_model': self.try_model,
            'force_exp': self.force_exp,
            'add_tags': self.add_tags,
            'search_spaces': self._get_search_spaces(),
            'all_fieldnames': self.all_fieldnames,
            'checked_fieldnames': self.checked_fieldnames,
        }

        if self.script_mode:
            return {**core_setting, **all_mode_settings}
        else:
            return {**core_setting, **all_mode_settings, **non_script_mode_settings}

    def get_csv_file_path(self, task_name, _process_index=0, _jump_data=False):
        """
        get the path of the specific data
        """
        if _jump_data:
            # get csv file name for jump data
            csv_file_name = self.jump_csv_file
            csv_file_path = os.path.join(self.root_path, self.data_dir, csv_file_name)
        else:
            if _process_index == 0:
                # get csv file name for core process
                csv_file_name = self.data_csv_file
            else:
                # get csv file name for other processes
                csv_file_name = self.data_csv_file_format.format(_process_index)
            csv_file_path = os.path.join(self.root_path, self.data_dir, task_name, csv_file_name)

        return csv_file_path

    # noinspection DuplicatedCode
    def output_script(self, _data):
        # get all possible parameters
        parameters = self._get_parameters()

        # filter the parameters according to the model
        model_parameters = {}
        for parameter in parameters:
            model = parameter['model']
            if model not in model_parameters:
                model_parameters[model] = []
            model_parameters[model].append(parameter)

        # save the script of each model
        for model in model_parameters:
            model_parameter = model_parameters[model]

            # filter the parameters according to the task name
            task_parameters = {}
            for parameter in model_parameter:
                task_name = parameter['task_name']
                if task_name not in task_parameters:
                    task_parameters[task_name] = []
                task_parameters[task_name].append(parameter)

            for task in task_parameters:
                parameters = task_parameters[task]

                # get the path of the specific script
                script_path = os.path.join(self.root_path, 'scripts', task, f'{_data}_script')

                # create the folder of the specific script
                if not os.path.exists(script_path):
                    os.makedirs(script_path)

                # get the time
                t = time.localtime()
                _run_time = time.strftime('%Y-%m-%d %H:%M:%S', t)

                # write the script of the same model and same task
                script_file = f'{script_path}/{model}.sh'
                if not os.path.exists(script_file):
                    with open(script_file, 'w') as f:
                        # write the header of the script
                        f.write(f'# This script is created by hyper-parameter optimizer at {_run_time}.\n')
                        f.write('\n')
                        f.write('export CUDA_VISIBLE_DEVICES=1\n')
                        f.write('\n')
                        f.write('model_name=' + f'{model}' + '\n')
                        f.write('\n')
                        # write the content of the script
                        for parameter in parameters:
                            f.write(f'# This segment is writen at {_run_time}.\n')
                            f.write('python -u run.py \\\n')
                            for key in parameter:
                                if key == 'model':
                                    f.write('\t--model $model_name\\\n')
                                f.write(f'\t--{key} {parameter[key]} \\\n')
                            f.write('\n')
                else:
                    # write the content of the script
                    with open(script_file, 'a') as f:
                        for parameter in parameters:
                            f.write(f'# This segment is writen at {_run_time}.\n')
                            f.write('python -u run.py \\\n')
                            for key in parameter:
                                if key == 'model':
                                    f.write('\t--model $model_name\\\n')
                                f.write(f'\t--{key} {parameter[key]} \\\n')
                            f.write('\n')

        # print the info of the successful output
        print(f'We successfully output the scripts in scripts folder!')

    def get_all_task_names(self):
        if self._task_names is not None:
            return self._task_names

        parameters = self._get_parameters()

        # get all task names
        task_names = []
        for parameter in parameters:
            task_name = parameter['task_name']
            if task_name not in task_names:
                task_names.append(task_name)

        self._task_names = task_names
        return task_names

    def start_search(self, process_index=0, force_test=False, inverse_exp=False, shutdown_after_done=False, prepare_config_params=None):
        # set default font to Times New Roman when plotting
        set_times_new_roman_font()
        print('==========================')
        # run directly under script mode
        if self.script_mode:
            # print info
            print('Hyper-parameter optimizer starts searching under script mode!')
            if shutdown_after_done:
                print(Fore.RED + 'Warning: System will shutdown after done!')
            print()

            print(Fore.BLUE + f'Start preparing config to args')
            # parse launch parameters and load default config
            args = self.prepare_config(prepare_config_params, True)

            print(f'{Fore.BLUE}Finished config. Build config dict')
            # create a dict to store the configuration values
            config = self._build_config_dict(args)

            # start experiment
            experiment_result = self._start_experiment(args, None, config, False, False, False)

            # phase criteria and save data
            self._save_experiment(config, experiment_result)

            # shutdown after done
            if shutdown_after_done:
                self._shutdown()

            return

        # print info
        print('Hyper-parameter optimizer starts searching under non-script mode!')
        print(f'Process index: {process_index}, Inverse experiments: {inverse_exp}.')
        if shutdown_after_done:
            print(Fore.RED + 'Warning: System will shutdown after done!')
        print()

        # check the index of the process
        if process_index > self.max_process_index or process_index < 0:
            raise ValueError(f'The index of the process {process_index} is out of range!')

        # check data header
        self._check_data_header()

        # init the config list
        config_list = []
        for task_name in self.get_all_task_names():
            # data files are under './data/{task_name}'
            # init the name of data file
            _csv_file_path = self.get_csv_file_path(task_name, _process_index=process_index)

            # init the head of data file
            self._init_header(_csv_file_path)

            # load config list in data file
            if process_index == 0:
                task_config_list = self._get_config_list(task_name, _csv_file_path, scan_all_csv=self.scan_all_csv)
            else:
                add_csv_file_path = self.get_csv_file_path(task_name, _process_index=0)
                task_config_list = self._get_config_list(task_name, [_csv_file_path, add_csv_file_path],
                                                         scan_all_csv=self.scan_all_csv)

            # combine config list
            config_list.extend(task_config_list)

        # jumped data files are under root data path
        # init the name of jumped data file
        _jump_csv_file_path = self.get_csv_file_path(None, _process_index=process_index, _jump_data=True)
        # init the head of jumped data file
        self._init_header(_jump_csv_file_path)
        # load config list in jumped data file
        jump_config_list = self._get_config_list(None, _jump_csv_file_path)

        # get all possible parameters
        parameters = self._get_parameters()

        # filter combinations with the known rules or trying models
        filtered_parameters = self._filter_parameters(parameters, jump_config_list, config_list, _jump_csv_file_path,
                                                      process_index, try_model=self.try_model,
                                                      force_exp=self.force_exp)

        # inverse the experiments if needed
        if inverse_exp:
            filtered_parameters = filtered_parameters[::-1]

        # equally distribute the parameters according to the number of processes
        # parameters = parameters[_process_index::(max_process_index + 1)]: It's in the order of the loops.
        # It's in the order in which they are arranged.
        process_parameters = self._distribute_parameters(filtered_parameters, process_index)

        # find total times
        total_times = len(process_parameters)
        print(f'Start total {total_times} experiments:\n')

        # iterate through the combinations and start searching by enumeration
        _time = 1
        finish_time = 0
        for parameter in process_parameters:
            # parse launch parameters and load default config
            args = self.prepare_config(parameter)

            # create a dict to store the configuration values
            config = self._build_config_dict(args)

            # link data of fieldnames with other fieldnames
            if self.link_fieldnames_data is not None:
                config = self.link_fieldnames_data(config)

            # set args for later
            args = self.set_args(args, config)

            # start experiment
            experiment_result = self._start_experiment(args, parameter, config, False, force_test,
                                                       (process_index == 0 and _time == 1))

            # phase criteria and save data
            self._save_experiment(config, experiment_result)

            print(f'>>>>>>> We have finished {_time}/{total_times}! >>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            finish_time = finish_time + 1
            _time = _time + 1

        print(f"We have finished {finish_time} times, {total_times} times in total!\n")

        # shutdown after done
        if shutdown_after_done:
            self._shutdown()

    def _get_search_spaces(self):
        if self.search_spaces is not None:
            return self.search_spaces

        search_spaces = {}
        for model in self.models:
            # get config list and model configs
            config_list, model_configs = self.get_search_space()

            # get default config
            config = {}
            for _config in config_list:
                for key in _config:
                    if key in config and config[key] != _config[key]:
                        raise ValueError(f'The key {key} has different values in the config list!')
                    config[key] = _config[key]

            # get config for specific model
            model_config = model_configs[model] if model_configs.get(model) else {}
            model_config['model'] = {'_type': 'single', '_value': model}

            # combine default config and model config
            for key, value in model_config.items():
                config[key] = value

            search_spaces[model] = config

        self.search_spaces = search_spaces
        return search_spaces

    def _get_parameters(self):
        if self._parameters is not None:
            return self._parameters

        _parameters = []
        for model in self.models:
            search_space = self._get_search_spaces()[model]

            # build parameters to be optimized from _search_space
            _params = {}
            for key in search_space:
                _params[key] = None  # _search_space[key]['_value']

            # get range of parameters for parameters from _search_space
            _parameters_range = {}
            for key in search_space:
                if search_space[key]['_type'] == 'single':
                    _parameters_range[key] = [search_space[key]['_value']]
                elif search_space[key]['_type'] == 'choice':
                    _parameters_range[key] = search_space[key]['_value']
                else:
                    raise ValueError(f'The type of {key} is not supported!')

            # generate all possible combinations of parameters within the specified ranges
            _combinations = list(product(*[_parameters_range[param] for param in _params]))

            # invert combinations to parameters and collect them
            for combination in _combinations:
                parameter = {param: value for param, value in zip(_params.keys(), combination)}
                _parameters.append(parameter)

        self._parameters = _parameters
        return _parameters

    def _filter_parameters(self, _parameters, _jump_config_list, _config_list, _jump_csv_file_path,
                           _process_index, try_model=True, force_exp=False, print_info=True):
        if print_info:
            print(f"We are filtering the parameters, please wait util it done to start other processes!")

            if force_exp:
                print(f'We are forced to run the experiments that we have finished in the csv data!')

        jump_time = 0
        filtered_parameters = []

        # if we need to try models, and then show the progress bar
        if try_model is True:
            _parameters = tqdm(_parameters)

        for parameter in _parameters:
            # check if we need to jump this experiment according to the known rules
            if self.check_jump_experiment is not None and self.check_jump_experiment(parameter):
                continue

            # parse launch parameters and load default config
            args = self.prepare_config(parameter)

            # create a dict to store the configuration values
            config = self._build_config_dict(args)

            # link data of fieldnames with other fieldnames
            if self.link_fieldnames_data is not None:
                config = self.link_fieldnames_data(config)

            # set args for later
            args = self.set_args(args, config)

            if not force_exp:
                # check if the parameters of this experiment need to be jumped
                if self._check_config_data(config, _jump_config_list):
                    continue

                # check if the parameters of this experiment have been done
                if self._check_config_data(config, _config_list):
                    continue

            # check if the model of this experiment can work
            if _process_index == 0 and try_model:
                # check if the parameters of this experiment is improper
                model_can_work = self._start_experiment(args, parameter, config, _try_model=True, _force_test=False,
                                                        _check_folder=False)

                # if the model cannot work, and then add it to the jump data file
                if not model_can_work:
                    self._save_config_dict(_jump_csv_file_path, config)
                    jump_time = jump_time + 1
                    continue

            filtered_parameters.append(parameter)

        if jump_time > 0:
            print(f"We found improper parameters and add {jump_time} experiments into {_jump_csv_file_path}!\n")

        return filtered_parameters

    def _init_experiment(self, task_name):
        # fix random seed
        self._fix_random_seed()

        # build experiment
        if task_name == 'long_term_forecast':
            self.Exp = Exp_Long_Term_Forecast
        elif task_name == 'short_term_forecast':
            self.Exp = Exp_Short_Term_Forecast
        elif task_name == 'imputation':
            self.Exp = Exp_Imputation
        elif task_name == 'anomaly_detection':
            self.Exp = Exp_Anomaly_Detection
        elif task_name == 'probability_forecast':
            self.Exp = Exp_Probability_Forecast
        elif task_name == 'classification':
            self.Exp = Exp_Classification

    def _start_experiment(self, _args, _parameter, _config, _try_model, _force_test, _check_folder):
        """
        If try_model is True, we will just try this model:
            if this model can work, then return True.
        """
        # start time
        exp_start_time, exp_start_run_time = self._get_run_time()
        if not _try_model:
            print('>>>>>>>({}) start experiment<<<<<<<'.format(exp_start_run_time))

        # build the setting of the experiment
        exp_setting, exp_train_time = self.build_setting(self.root_path, _args, exp_start_run_time, self.time_format,
                                                         self.get_custom_test_time, _try_model)

        # get the experiment type
        self._init_experiment(_args.task_name)

        # try model if needed
        if _try_model:
            # build the experiment
            exp = self.Exp(self.root_path, _args, try_model=True, save_process=False)

            # validate the model
            valid, best_model_path = exp.train(exp_setting, check_folder=True)

            # return the results
            return valid

        if _args.is_training:
            # build the experiment
            exp = self.Exp(self.root_path, _args, try_model=False, save_process=self.save_process)
            
            # print info of the experiment
            if _parameter is not None:
                exp.print_content(f'Optimizing params in experiment:{_parameter}')
            exp.print_content(f'Config in experiment:{_config}')
            print_args(_args, exp.print_content)

            # start training
            _, exp_train_run_time = self._get_run_time()
            exp.print_content('>>>>>>>({}) start training: {}<<<<<<<'.format(exp_train_run_time, exp_setting))
            stop_epochs, best_model_path = exp.train(exp_setting, check_folder=_check_folder)
            exp.print_content(f"[_start_experiment] Best model path {best_model_path}")
            
            # start testing
            _, exp_test_run_time = self._get_run_time()
            exp.print_content('>>>>>>>({}) start testing: {}<<<<<<<'.format(exp_test_run_time, exp_setting))
            eva_config = exp.test(exp_setting, test=_force_test, check_folder=_check_folder)

            # clean cuda cache
            torch.cuda.empty_cache()
        else:
            # build the experiment
            exp = self.Exp(self.root_path, _args, try_model=False, save_process=self.save_process)

            # print info of the experiment
            if _parameter is not None:
                exp.print_content(f'Optimizing params in experiment:{_parameter}')
            exp.print_content(f'Config in experiment:{_config}')
            print_args(_args, exp.print_content)

            # start testing
            _, exp_test_run_time = self._get_run_time()
            exp.print_content('>>>>>>>({}) start testing: {}<<<<<<<'.format(exp_test_run_time, exp_setting))
            stop_epochs, best_model_path = exp.train(exp_setting, check_folder=_check_folder, only_init=True)
            exp.print_content(f"[_start_experiment] Best model path {best_model_path}")
            eva_config = exp.test(exp_setting, test=True, check_folder=_check_folder)

            # clean cuda cache
            torch.cuda.empty_cache()

        # end experiment
        exp_end_time, exp_end_run_time = self._get_run_time()
        exp_time = self._get_diff_time(exp_start_time, exp_end_time)
        print('total cost time: {}'.format(exp_time))
        print('>>>>>>>({}) end experiment<<<<<<<'.format(exp_end_run_time))

        return eva_config, exp_train_time, exp_setting, stop_epochs

    def _get_run_time(self):
        current_time = time.localtime()
        run_time = time.strftime(self.time_format, current_time)
        return current_time, run_time

    def _get_diff_time(self, start_time, end_time):
        return time.strftime(self.diff_time_format, time.gmtime(time.mktime(end_time) - time.mktime(start_time)))

    def _save_experiment(self, config, _experiment_result):
        # unpack the experiment result
        eva_config, exp_start_run_time, setting, stop_epochs = _experiment_result

        # phase criteria and save data
        if eva_config is not None:
            # phase criteria data from eva_config
            mse = eva_config.get('mse', None)
            mae = eva_config.get('mae', None)
            acc = eva_config.get('acc', None)
            smape = eva_config.get('smape', None)
            f_score = eva_config.get('f_score', None)
            crps = eva_config.get('crps', None)
            mre = eva_config.get('mre', None)
            pinaw = eva_config.get('pinaw', None)

            # load criteria data
            config['mse'] = mse
            config['mae'] = mae
            config['acc'] = acc
            config['smape'] = smape
            config['f_score'] = f_score
            config['crps'] = crps
            config['mre'] = mre
            config['pinaw'] = pinaw

            # load setting and run time
            config['setting'] = setting
            config['run_time'] = exp_start_run_time
            config['stop_epochs'] = stop_epochs

            _csv_file_path = self.get_csv_file_path(config['task_name'])
            self._save_config_dict(_csv_file_path, config)

    def _fix_random_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _build_config_dict(self, _args):
        config_dict = self.build_config_dict(_args)
        config_dict['seed'] = self.seed
        config_dict['model_id'] = config_dict['model_id'] + self._get_tags(_args, self.add_tags)

        return config_dict

    def _get_tags(self, _args, _add_tags):
        tags = []

        if self.get_model_id_tags is not None:
            tags = self.get_model_id_tags(_args)

        for add_tag in _add_tags:
            tags.append(add_tag)

        if len(tags) == 0:
            return ''
        else:
            tags_text = ''
            for label in tags:
                tags_text = tags_text + label + ', '
            tags_text = tags_text[:-2]
            return f'({tags_text})'

    def _init_header(self, file_path):
        # check the folder path
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if not os.path.exists(file_path):
            # create new file with header
            with open(file_path, 'w', newline='') as csv_file:
                _writer = csv.DictWriter(csv_file, fieldnames=self.all_fieldnames)
                _writer.writeheader()

    def _check_header_correct(self, header):
        print(f'Start checking header correctness {header} -> {self.all_fieldnames}')
        correct = True
        exist_headers = []

        if len(header) != len(self.all_fieldnames):
            correct = False
            for i in range(len(header)):
                if header[i] in self.all_fieldnames:
                    exist_headers.append(header[i])
        else:
            for i in range(len(header)):
                if header[i] != self.all_fieldnames[i]:
                    correct = False
                if header[i] in self.all_fieldnames:
                    exist_headers.append(header[i])
        return correct, exist_headers

    def _check_data_header(self):
        
        root_path = os.path.join(self.root_path, self.data_dir)
        print(f'check data header ... {root_path}')
        
        # get all csv file under the path
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)

                    new_data = None

                    # check the header of the file
                    with open(file_path, 'r', newline='') as csv_file:
                        file_header = next(csv.reader(csv_file))

                        correct, exist_headers = self._check_header_correct(file_header)
                        if not correct:
                            # get old data
                            old_data = []
                            _reader = csv.DictReader(csv_file, fieldnames=file_header)
                            for row in _reader:
                                old_data.append(row)

                            # transfer exist data
                            new_data = []
                            for _dict in old_data:
                                row = {}
                                for exist_header in exist_headers:
                                    row[exist_header] = _dict[exist_header]
                                new_data.append(row)
                            
                            # create default new data
                            non_exist_headers = list(set(self.all_fieldnames) - set(exist_headers))
                            print(f'non exist headers : {non_exist_headers}')
                            default_args = self.prepare_config(None, False)
                            print(f'default args : {default_args}')
                            default_data_dict = self._build_config_dict(default_args)
                            print(f'Finished building default data dict : {default_data_dict}')
                            print('--------------------------')
                            for row in new_data:
                                print(f'row : {row}')
                                for non_exist_header in non_exist_headers:
                                    if non_exist_header in default_data_dict:
                                        print(f'non_exist_header = {non_exist_header} -> {default_data_dict[non_exist_header]}')
                                        row[non_exist_header] = default_data_dict[non_exist_header]
                    print(f'Finished stuff')
                    # write new data if not correct
                    if new_data is not None:
                        # create new file with header
                        with open(file_path, 'w', newline='') as csv_file:
                            _writer = csv.DictWriter(csv_file, fieldnames=self.all_fieldnames)
                            _writer.writeheader()
                            for row in new_data:
                                _writer.writerow(row)

    def _get_config_list(self, task_name, file_paths, scan_all_csv=False):
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        if scan_all_csv:
            root_path = os.path.join(self.root_path, self.data_dir, task_name)
            # get all csv file under the path
            for root, dirs, files in os.walk(root_path):
                for file in files:
                    if file == self.jump_csv_file:
                        continue
                    if file.endswith('.csv') and file not in file_paths:
                        _append_path = os.path.join(root, file)
                        self._init_header(_append_path)
                        file_paths.append(_append_path)

        _config_list = []
        for file_path in file_paths:
            _ = []
            with open(file_path, 'r') as csv_file:
                reader = csv.DictReader(csv_file, fieldnames=self.all_fieldnames)
                next(reader)  # skip the header
                for row in reader:
                    _.append(row)
            _config_list.extend(_)

        return _config_list

    def _check_config_data(self, config_data, _config_list):
        for _config in _config_list:
            flag = True
            for _field in self.checked_fieldnames:
                if not self._check_data_same(_config[_field], config_data[_field]):
                    flag = False
                    break
            if flag:
                return True
        return False

    @staticmethod
    def _check_data_same(_data1, _data2):
        # check None value
        if _data1 is None and _data2 == '':
            return True
        if _data1 == '' and _data2 is None:
            return True

        # check string value
        str1 = str(_data1)
        str2 = str(_data2)
        if str1 == str2:
            return True

        # check number value
        # noinspection PyBroadException
        try:
            value1 = eval(str1)
            value2 = eval(str2)
            if value1 == value2:
                return True
        except:
            return False

        return False

    def _save_config_dict(self, file_path, _config):
        logger.info(f'check folder etc for file: {file_path}')
        folder = os.path.dirname(file_path)
        logger.info(f'check if folder {folder} exists')
        if not os.path.exists(folder):
            logger.info(f'folder {folder} didnt exit. Create it')
            os.makedirs(folder)
        
        # delete the fieldnames in _config that not in _fieldnames
        for key in list(_config.keys()):
            if key not in self.all_fieldnames:
                del _config[key]
        
        with open(file_path, 'a', newline='') as csvfile:
            _writer = csv.DictWriter(csvfile, fieldnames=self.all_fieldnames)
            _writer.writerow(_config)

    def _distribute_parameters(self, _parameters, _process_index):
        # Calculate the number of parameters per process
        processes_number = self.max_process_index + 1
        combinations_per_process = len(_parameters) // processes_number
        remainder = len(_parameters) % processes_number

        # Initialize variables to keep track of the current index
        current_index = 0

        # Iterate over each process
        for i in range(processes_number):
            # Calculate the start and end indices for the current process
            start_index = current_index
            end_index = start_index + combinations_per_process + (1 if i < remainder else 0)

            # Get the parameters for the current process
            process_parameters = _parameters[start_index:end_index]

            # Update the current index for the next process
            current_index = end_index

            # Return parameters for the current process
            if i == _process_index:
                return process_parameters

    @staticmethod
    def _shutdown():
        import platform
        system = platform.system()

        if system == 'Windows':
            os.system('shutdown -s -t 0')
        elif system == 'Linux':
            os.system("shutdown -t now")
        else:
            print(f'The system {system} is not supported to shutdown!')
