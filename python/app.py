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
import requests
import sys
import logging
from utils.downloaddata import BlobRelatedClass
from flask_dapr.app import DaprApp
from dapr.clients import DaprClient
from hyper_parameter_optimizer.optimizer import HyperParameterOptimizer


app = flask.Flask(__name__)
app.logger.setLevel(logging.DEBUG)
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
            data=json.dumps({"success": True})
        )
    
    app.logger.info(f'[download_data_from_azure]Finished sending message back to orchestrator')
    return "success", 200

@dapr_app.subscribe(AI_PUBSUB, 'start-train-model')
def start_train_model():
    app.logger.info(f'[start_train_model] Start Training model')
    app.logger.info(f'[start_train_model] Finished training model. Send message back to orchestrator')
    
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
        'gpu': f'{gpu}'    ,
        
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
    app.logger.info(f'Start search and training model ... ')
    h.start_search(prepare_config_params=prep_configs)
    app.logger.info('Finished start search on finding model ')
    result = {
        "success" : True
    }
    strResult = json.dumps(result)
    app.logger.info(f'Returning : {strResult}')
    
    with DaprClient() as client:
        result = client.publish_event(
            pubsub_name='ai-pubsub',
            topic_name='finished-train-model',
            data=strResult
        )
        app.logger.info(f'[start_train_model] result ; {result}')
    app.logger.info(f'[start_train_model] Finished sending message back to orchestrator')
    return "success", 200

# @app.route('/randomNumber', methods=['GET'])
# def random_number():
#     return jsonify(random.randint(0, 101))

# @app.route('/saveNumber', methods=['POST'])
# def save_number():
#     content = request.json
#     number = content['number']
#     response = requests.post(state_url, json=[{"key": "savedNumber", "value": number}])
#     print(response, flush="true")
#     return "OK"

# @app.route('/savedNumber', methods=['GET'])
# def get_saved_number():
#     response=requests.get(f'{state_url}/savedNumber')
#     return json.dumps(response.json()), 200, {'ContentType':'application/json'} 

# @app.route('/dapr/subscribe', methods=['GET'])
# def subscribe():
#     return jsonify(["A", "B"])

# @app.route('/A', methods=['POST'])
# def topicAHandler():
#     print(f'A: {request.json}', flush=True)
#     return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

# @app.route('/B', methods=['POST'])
# def topicBHandler():
#     print(f'B: {request.json}', flush=True)
#     return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

app.run(host='0.0.0.0', port=5001)