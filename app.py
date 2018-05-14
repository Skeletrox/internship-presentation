from filter import create_data
from tf_interface import tf_classify_train, tf_classify_predict
from flask import Flask
from flask import request
from flask import Response
from flask_cors import CORS, cross_origin
import json


app = Flask(__name__)


@app.route('/train', methods=['GET'])
def train():
    result = create_data(False)
    classify_result = tf_classify_train()
    response = {
        'success': result == 0,
        'message': 'Created training data' if classify_result == 0 else 'Error'
    }
    js = json.dumps(response)
    r = Response(js, status=200, mimetype='application/json')
    return r


@app.route('/classify', methods=['GET'])
def classify():
    result = create_data(True)
    classify_result = tf_classify_predict()
    response = {
        'success': result == 0,
        'data': classify_result,
    }
    js = json.dumps(response)
    r = Response(js, status=200, mimetype='application/json')
    return r
