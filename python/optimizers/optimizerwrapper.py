
from hyper_parameter_optimizer.optimizer import HyperParameterOptimizer
import os

class OptimizerWrapper(object):
    def __init__(self, is_training: int = 1) -> None:
        seq_len = 96
        enc_in=4
        c_out=1
        d_model=16
        e_layers=2
        d_ff=32
        down_sampling_layers=3
        down_sampling_window=2
        train_epochs=int(os.getenv('train_epochs',50))
        batch_size=int(os.getenv('batch_size', 128))
        patience=int(os.getenv('patience', 5)) #10
        learning_rate=0.01
        use_gpu=1
        gpu=0
        lstm_hidden_size=512
        lstm_layers=1

        self.prep_configs = {
            'task_name': 'long_term_forecast',
            'is_training': is_training,
            'model_id':f'LSTM_{seq_len}_96',
            'model': 'LSTM',
            'data': 'custom',
            'root_path': './data/',
            'data_path':'merged_and_sorted_file.csv',
            'features':'M',
            'target':'Watt',
            'scaler':'StandardScaler',
            'seq_len':seq_len,
            'label_len':0,
            'pred_len':96,
            'enc_in':enc_in,
            'c_out': c_out,
            'd_model':d_model,
            'e_layers':e_layers,
            'd_ff':d_ff,
            'down_sampling_layers':down_sampling_layers,
            'down_sampling_window':down_sampling_window,
            'down_sampling_method': 'avg',
            'train_epochs': train_epochs,
            'batch_size': batch_size,
            'patience': patience,
            'learning_rate': learning_rate,
            'des': 'Exp',
            'use_gpu': use_gpu,
            'gpu': gpu,
            
            'devices': '0',
            'freq': 'h',
            'lag': 0,
            'reindex': 0,
            'reindex_tolerance': 0.9,
            'pin_memory': True,
            'seasonal_patterns' : 'Monthly',
            'inverse': False,
            'mask_rate' : 0.25,
            'anomaly_ratio' : 0.25,
            'expand': 2,
            'd_conv' : 4,
            'top_k': 5,
            'num_kernels' : 6,
            'dec_in': 7,
            'n_heads': 8,
            'd_layers': 1,
            'd_ff': 2048,
            'moving_avg': 25,
            'series_decomp_mode': 'avg',
            'factor': 1,
            'distil': False,
            'dropout': 0.1,
            'embed': 'timeF',
            'activation': 'gelu',
            'output_attention': True,
            'channel_independence': 1,
            'decomp_method': 'moving_avg',
            'use_norm': 1,
            'seg_len': 48,
            'num_workers': 12,
            'itr': 1,
            'loss': 'auto',
            'lradj': 'type1',
            'use_amp': False,
            'p_hidden_dims': [128,128],
            'p_hidden_layers': 2,
            'use_dtw': False,
            'augmentation_ratio': 0,
            'jitter': False,
            'scaling': False,
            'permutation': False,
            'randompermutation': False,
            'magwarp': False,
            'timewarp': False,
            'windowslice': False,
            'windowwarp': False,
            'rotation': False,
            'spawner': False,
            'dtwwarp': False,
            'shapedtwwarp': False,
            'wdba': False,
            'discdtw': False,
            'discsdtw': False,
            'extra_tag': '',
            'lstm_hidden_size': 512,
            'lstm_layers': 1,
            'num_spline': 20,
            'sample_times': 99,
            'custom_params': ''
            
        }
    
    def build_setting(self, root_path, args, exp_start_run_time,time_format,   get_custom_test_time, _try_model):
        return "models", ''
    
    def start_testing(self):
        h = HyperParameterOptimizer(script_mode=True, build_setting=self.build_setting )
        h.config_optimizer_settings(root_path='.',
                                    data_dir='data',
                                    jump_csv_file='jump_data.csv',
                                    data_csv_file='merged_and_sorted_file.csv',
                                    data_csv_file_format='merged_and_sorted_file_{}.csv',
                                    scan_all_csv=True,
                                    process_number=1,
                                    save_process=True)
        
        h.start_search(prepare_config_params=self.prep_configs)
    
    def startTraining(self):
        h = HyperParameterOptimizer(script_mode=True )
        h.config_optimizer_settings(root_path='.',
                                data_dir='data',
                                jump_csv_file='jump_data.csv',
                                data_csv_file='merged_and_sorted_file.csv',
                                data_csv_file_format='merged_and_sorted_file_{}.csv',
                                scan_all_csv=True,
                                process_number=1,
                                save_process=True)
        
        h.start_search(prepare_config_params=self.prep_configs)