# ------------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------------------------------
import traceback    
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
from optimizers.optimizerwrapper import OptimizerWrapper
from utils.downloaddata import BlobRelatedClass
from flask_dapr.app import DaprApp
from dapr.clients import DaprClient
from hyper_parameter_optimizer.optimizer import HyperParameterOptimizer
from logging.config import dictConfig
import matplotlib
from utils.singleton import SingletonClass

from flasgger import Swagger


swagger_destination_path = '/static/swagger.yaml'




matplotlib.pyplot.set_loglevel(level ="error")


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
        "root": {"level": "WARNING", "handlers": ["console", "time-rotate"]},
        'app': {'level': 'DEBUG', "handlers": ['console', 'time-rotate']},
        "werkzeug": {  # Flask
            "handlers": ["console", " time-rotate"],
            "level": logging.WARNING,
            "propagate": False,
        },
        "matplotlib": {  # Flask
            "handlers": ["console", "time-rotate"],
            "level": logging.WARNING,
            "propagate": False,
        },
        "numba.core.ssa": {  # Flask
            "handlers": ["console", "time-rotate"],
            "level": logging.ERROR,
            "propagate": False,
        },
        
    }
)



logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.backends.backend_pdf').setLevel(logging.WARNING)
logging.getLogger('numba.core.ssa').setLevel(logging.WARNING)

app = flask.Flask(__name__, static_url_path='/static', static_folder='./static')
swagger = Swagger(app)

dapr_app = DaprApp(app)

#CORS(app)

AI_PUBSUB='ai-pubsub'

#dapr_port = os.getenv("DAPR_HTTP_PORT", 3500)
#state_url = "http://localhost:{}/v1.0/state".format(dapr_port)

@app.route('/healtz', methods=['GET'])
def health():
    """Example Healthz endpoint
    This endpoint is being used as an healthz probe for k3s
    ---
    definitions:
      Success:
        type: object
        properties:
          success:
            type: boolean
      NonSuccess:
        type: object
        properties:
          type: boolean
            
    responses:
      200:
        description: success is true => Everything is up and running
        schema:
          $ref: '#/definitions/Success'
        examples:
          success: true
    """
    return jsonify({'success' : True})

@app.route('/train_model', methods=['GET'])
def train_model():
    """
    Start Training model
    This is a dummy method. It should not be used
    ---
    responses:
      200:
        description: This just returns success true
        schema: 
          $ref: '#/definitions/Success'
        examples:
          success: true
    """
    app.logger.info(f'Start Training model on data')
    objectToReturn = {
        "success" : True
    }
    return jsonify(objectToReturn)


@dapr_app.subscribe(AI_PUBSUB, 'start-download-data')
@app.route('/download_data', methods=['GET'])
def download_data_from_azure( ):
    """
        Download data from azure
        Download training data from azure. So we can train our model based on this.
        This method is being called from external applications through Dapr 
        ---
        responses:
          200:
            description: Once data has been downloaded. Return success object
            schema: 
                $ref: '#/definitions/Success'
            examples:
                success: true
              
    """
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

@app.route('/test_model', methods=['GET'])
def start_test_model():
    """
    Test latest trained AI Model
    This method is being used to run a test on the latest AI model we trained.
    It returns the results in images, being stored on the shared storage. When
    the training / testing is already running. Ignore this request and just 
    return success true
    ---
    responses:
        200:
            description: Training was successfull
            schema: 
                $ref: '#/definitions/Success'
            examples:
                success: True
        500:
            description: There was an issue with testing the latest model
            schema:
                $ref: '#/definitions/Success'
            examples: 
                success: False
    """
    if singleton.isRunning:
        app.logger.info(f'[start_test_model] Instance was already running... So cant start Test')
        return "success", 200
    else:
        singleton.isRunning = True
    
    app.logger.info(f'[start_test_model] Start testing model')
    try: 
        optimizer = OptimizerWrapper(is_training=0)
        optimizer.start_testing()
    except Exception as error:
        app.logger.error(f'There was an error testing {error}')
        app.logger.error(traceback.format_exc())
        singleton.isRunning = False
        return jsonify({'success' : False}), 500
    return "success", 200

@dapr_app.subscribe(AI_PUBSUB, 'start-train-model')
@app.route('/start_train_model', methods=['GET'])
def start_train_model():
    """
    Start Training AI Model 
    Start the training of the model based on the downloaded data.
    When the training / testing is already running. Ignore this request and just 
    return success true
    ---
    responses:
        200:
            description: Training was successfull
            schema: 
                $ref: '#/definitions/Success'
            examples:
                success: True
        500:
            description: There was an issue with testing the latest model
            schema:
                $ref: '#/definitions/Success'
            examples: 
                success: False
    """
    if singleton.isRunning: 
        app.logger.info(f'[start_train_model] Instane was already running ... so just return ok')
        return "success", 200
    else:
        singleton.isRunning = True
        
    app.logger.info(f'[start_train_model] Start Training model')
    perform_training = os.getenv('perform_training', "False").lower() == "true"
    
    training_result = True
    try:
        app.logger.info(f'[start_train_model] Actually perform training : {perform_training}')
        if perform_training:
            optimizerWrapper = OptimizerWrapper(is_training=(1 if perform_training else 0))
            optimizerWrapper.startTraining()
        app.logger.info(f'Start Search finished. And no exception was thrown')
    except Exception as e:
        app.logger.error('there was an issue training data ... ',exc_info=True)
        app.logger.info(e)
        training_result = False
    
    
    app.logger.info('Finished start search on finding model ')
    result = {
        "success" : training_result
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
    return jsonify({'success' : training_result}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)