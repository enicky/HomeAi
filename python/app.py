# ------------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------------------------------

import flask
from flask import request, jsonify, Response
from flask_cors import CORS
import json
import math
import os
import random
import matplotlib.pyplot
import requests
import sys
import logging
from utils.downloaddata import BlobRelatedClass
from flask_dapr.app import DaprApp
from dapr.clients import DaprClient
from hyper_parameter_optimizer.optimizer import HyperParameterOptimizer
from logging.config import dictConfig
import matplotlib

matplotlib.pyplot.set_loglevel(level ="error")


class SingletonClass(object):
    isRunning = False
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SingletonClass, cls).__new__(cls)
        return cls.instance

singleton = SingletonClass()

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "time-rotate": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": "flask.log",
                "when": "D",
                "interval": 10,
                "backupCount": 5,
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "time-rotate"]},
    }
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.backends.backend_pdf').setLevel(logging.WARNING)
logging.getLogger('numba.core.ssa').setLevel(logging.WARNING)

app = flask.Flask(__name__)
dapr_app = DaprApp(app)

#CORS(app)

AI_PUBSUB='ai-pubsub'

#dapr_port = os.getenv("DAPR_HTTP_PORT", 3500)
#state_url = "http://localhost:{}/v1.0/state".format(dapr_port)

@app.route('/healtz', methods=['GET'])
def health():
    return jsonify({'success' : True})
@app.route('/train_model', methods=['GET'])
def train_model():
    app.logger.info(f'Start Training model on data')
    objectToReturn = {
        "success" : True
    }
    return jsonify(objectToReturn)


#@app.route('/download_data', methods=['GET'])
@dapr_app.subscribe(AI_PUBSUB, 'start-download-data')
def download_data_from_azure( ):
    app.logger.info(f'[download_data_from_azure] Downloading data from azure to process ')
    b = BlobRelatedClass()
    b.start_downloading_data()
    app.logger.info(f'[download_data_from_azure]Finished sending downloading. Start sending message back to orchestrator')
    with DaprClient() as client:
        result = client.publish_event(
            pubsub_name='ai-pubsub',
            topic_name="finished-download-data",
            data=json.dumps({"success": True, "canStartTraining": True})
        )
    
    app.logger.info(f'[download_data_from_azure]Finished sending message back to orchestrator')
    return "success", 200

@dapr_app.subscribe(AI_PUBSUB, 'start-train-model')
def start_train_model():
    if singleton.isRunning: 
        app.logger.info(f'[start_train_model] Instane was already running ... so just return ok')
        return "success", 200
    else:
        singleton.isRunning = True
        
    app.logger.info(f'[start_train_model] Start Training model')
    app.logger.info(f'[start_train_model] Finished training model. Send message back to orchestrator')
    perform_training = os.getenv('perform_training', "False").lower() == "true"
    app.logger.info(f'[start_train_model] Actually perform training : {perform_training}')
        
    
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
    train_epochs=int(os.getenv('train_epochs',50))
    batch_size=int(os.getenv('batch_size', 128))
    patience=int(os.getenv('patience', 5)) #10
    learning_rate=0.01
    use_gpu=1
    gpu=0
    lstm_hidden_size=512
    lstm_layers=1

    prep_configs = {
        'task_name': 'long_term_forecast',
        'is_training': 1,
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
    app.logger.info(f'Start search and training model ... {perform_training}')
    try:
        if perform_training:
            h.start_search(prepare_config_params=prep_configs)
        app.logger.info(f'Start Search finished. And no exception was thrown')
    except Exception as e:
        app.logger.error('there was an issue training data ... ',exc_info=True)
        app.logger.info(e)
    
    
    app.logger.info('Finished start search on finding model ')
    result = {
        "success" : True
    }
    strResult = json.dumps(result)
    app.logger.info(f'Returning : {strResult}')
    
    with DaprClient() as client:
        result = client.publish_event(
            pubsub_name=AI_PUBSUB,
            topic_name='finished-train-model',
            data=strResult
        )
        app.logger.info(f'[start_train_model] result ; {result}')
    app.logger.info(f'[start_train_model] Finished sending message back to orchestrator')
    singleton.isRunning = False
    app.logger.info('isRunning was set to false => Can start processing again!')
    return "success", 200

app.run(host='0.0.0.0', port=5001)