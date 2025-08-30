from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import joblib
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        columns = df.columns.tolist()
        dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
        
        return jsonify({
            'filename': filename,
            'columns': columns,
            'dtypes': dtypes,
            'shape': df.shape,
            'preview': df.head(5).to_dict()
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/models', methods=['GET'])
def get_models():
    models = [
        {'id': 'lstm', 'name': 'LSTM', 'type': 'temporal'},
        {'id': 'cnn', 'name': 'CNN', 'type': 'spatial'},
        {'id': 'decision_tree', 'name': 'Árbol de Decisión', 'type': 'traditional'},
        {'id': 'random_forest', 'name': 'Random Forest', 'type': 'traditional'},
        {'id': 'transformer', 'name': 'Transformer', 'type': 'temporal'},
        {'id': 'gru', 'name': 'GRU', 'type': 'temporal'}
    ]
    return jsonify(models)

@app.route('/api/normalization-methods/<model_type>', methods=['GET'])
def get_normalization_methods(model_type):
    methods = {
        'lstm': ['MinMax', 'StandardScaler', 'RobustScaler', 'Normalizer'],
        'cnn': ['MinMax', 'StandardScaler', 'ImageNormalization'],
        'decision_tree': ['None', 'MinMax', 'StandardScaler'],
        'random_forest': ['None', 'MinMax', 'StandardScaler'],
        'transformer': ['LayerNorm', 'MinMax', 'StandardScaler'],
        'gru': ['MinMax', 'StandardScaler', 'RobustScaler']
    }
    return jsonify(methods.get(model_type, ['MinMax', 'StandardScaler']))

@app.route('/api/hyperparameters/<model_type>', methods=['GET'])
def get_hyperparameters(model_type):
    hyperparams = {
        'lstm': {
            'units': {'type': 'int', 'default': 50, 'min': 10, 'max': 500},
            'layers': {'type': 'int', 'default': 2, 'min': 1, 'max': 5},
            'dropout': {'type': 'float', 'default': 0.2, 'min': 0, 'max': 0.5},
            'learning_rate': {'type': 'float', 'default': 0.001, 'min': 0.0001, 'max': 0.1},
            'batch_size': {'type': 'int', 'default': 32, 'min': 16, 'max': 128},
            'epochs': {'type': 'int', 'default': 100, 'min': 10, 'max': 1000}
        },
        'cnn': {
            'filters': {'type': 'int', 'default': 32, 'min': 8, 'max': 256},
            'kernel_size': {'type': 'int', 'default': 3, 'min': 2, 'max': 7},
            'layers': {'type': 'int', 'default': 3, 'min': 1, 'max': 10},
            'dense_units': {'type': 'int', 'default': 128, 'min': 32, 'max': 512},
            'learning_rate': {'type': 'float', 'default': 0.001, 'min': 0.0001, 'max': 0.1},
            'batch_size': {'type': 'int', 'default': 32, 'min': 16, 'max': 128},
            'epochs': {'type': 'int', 'default': 50, 'min': 10, 'max': 500}
        },
        'decision_tree': {
            'max_depth': {'type': 'int', 'default': 10, 'min': 1, 'max': 50},
            'min_samples_split': {'type': 'int', 'default': 2, 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'default': 1, 'min': 1, 'max': 10},
            'criterion': {'type': 'select', 'options': ['gini', 'entropy'], 'default': 'gini'}
        },
        'random_forest': {
            'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 1000},
            'max_depth': {'type': 'int', 'default': 10, 'min': 1, 'max': 50},
            'min_samples_split': {'type': 'int', 'default': 2, 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'default': 1, 'min': 1, 'max': 10},
            'criterion': {'type': 'select', 'options': ['gini', 'entropy'], 'default': 'gini'}
        },
        'transformer': {
            'num_heads': {'type': 'int', 'default': 8, 'min': 2, 'max': 16},
            'num_layers': {'type': 'int', 'default': 4, 'min': 1, 'max': 12},
            'd_model': {'type': 'int', 'default': 256, 'min': 64, 'max': 1024},
            'dropout': {'type': 'float', 'default': 0.1, 'min': 0, 'max': 0.5},
            'learning_rate': {'type': 'float', 'default': 0.0001, 'min': 0.00001, 'max': 0.01},
            'batch_size': {'type': 'int', 'default': 32, 'min': 16, 'max': 128},
            'epochs': {'type': 'int', 'default': 100, 'min': 10, 'max': 1000}
        },
        'gru': {
            'units': {'type': 'int', 'default': 50, 'min': 10, 'max': 500},
            'layers': {'type': 'int', 'default': 2, 'min': 1, 'max': 5},
            'dropout': {'type': 'float', 'default': 0.2, 'min': 0, 'max': 0.5},
            'learning_rate': {'type': 'float', 'default': 0.001, 'min': 0.0001, 'max': 0.1},
            'batch_size': {'type': 'int', 'default': 32, 'min': 16, 'max': 128},
            'epochs': {'type': 'int', 'default': 100, 'min': 10, 'max': 1000}
        }
    }
    return jsonify(hyperparams.get(model_type, {}))

@app.route('/api/metrics/<model_type>', methods=['GET'])
def get_metrics(model_type):
    metrics = {
        'classification': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'confusion_matrix'],
        'regression': ['mse', 'rmse', 'mae', 'r2_score', 'mape']
    }
    
    if model_type in ['decision_tree', 'random_forest']:
        return jsonify({
            'classification': metrics['classification'],
            'regression': metrics['regression']
        })
    else:
        return jsonify(metrics['regression'])

@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.json
    
    return jsonify({
        'status': 'training',
        'message': 'Model training started',
        'job_id': datetime.now().timestamp()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    
    return jsonify({
        'predictions': [],
        'status': 'success'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)