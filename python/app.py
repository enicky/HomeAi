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



app = flask.Flask(__name__)

CORS(app)

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

@app.route('/download_data', methods=['GET'])
def download_data_from_azure():
    app.logger.info(f'Downloading data from azure to process')
    b = BlobRelatedClass()
    b.start_downloading_data()
    return jsonify({
        "success": True
    })

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