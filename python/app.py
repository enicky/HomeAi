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