from hyper_parameter_optimizer.optimizer import HyperParameterOptimizer

h = HyperParameterOptimizer(script_mode=True )

h.config_optimizer_settings(root_path='.',
                            data_dir='data',
                            jump_csv_file='jump_data.csv',
                            data_csv_file='merged_and_sorted_file.csv',
                            data_csv_file_format='merged_and_sorted_file_{}.csv',
                            scan_all_csv=True,
                            process_number=1,
                            save_process=True)

seq_len = 96
enc_in=4
c_out=1
d_model=16
e_layers=2
d_ff=32
down_sampling_layers=3
down_sampling_window=2
train_epochs=50
batch_size=128
patience=10
learning_rate=0.01
use_gpu=1
gpu=0
lstm_hidden_size=1
lstm_layers=2

prep_configs = {
    'task_name': 'long_term_forecast',
    'is_training': '1',
    'model_id':f'LSTM_{seq_len}_96',
    'model': 'LSTM',
    'data': 'custom',
    'root_path': './data/',
    'data_path':'merged_and_sorted_file.csv',
    'features':'M',
    'target':'Watt',
    'scaler':'StandardScaler',
    'seq_len':f'{seq_len}',
    'label_len':'0',
    'pred_len':'96',
    'enc_in':f'{enc_in}',
    'c_out':f'{c_out}',
    'd_model':f'{d_model}',
    'e_layers':f'{e_layers}',
    'd_ff':f'{d_ff}',
    'down_sampling_layers':f'{down_sampling_layers}',
    'down_sampling_window':f'{down_sampling_window}',
    'down_sampling_method': 'avg',
    'train_epochs': f'{train_epochs}',
    'batch_size': f'{batch_size}',
    'patience': f'{patience}',
    'learning_rate': f'{learning_rate}',
    'des': 'Exp',
    'use_gpu': f'{use_gpu}',
    'gpu': f'{gpu}'    
}

if __name__ == "__main__":
    h.start_search(prepare_config_params=prep_configs)
